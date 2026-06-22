import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.FileVisitor;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Reformat UCSF Brain Mets data into nnU-Net raw dataset layout.
 *
 * Output naming:
 * - imagesTr/UCSFbrainmets_[xxx]_0000.nii.gz (FLAIR)
 * - imagesTr/UCSFbrainmets_[xxx]_0001.nii.gz (T1pre)
 * - imagesTr/UCSFbrainmets_[xxx]_0002.nii.gz (T1post)
 * - imagesTr/UCSFbrainmets_[xxx]_0003.nii.gz (T2Synth)
 * - labelsTr/UCSFbrainmets_[xxx].nii.gz      (BraTS-seg label)
 *
 * Usage:
 *   java reformat [sourceRoot] [targetDatasetRoot]
 *
 * Defaults:
 * - sourceRoot:
 *   /mnt/d/A1_RainSun_20240916/1-UWMadison/IDiA-Lab/Medical_Images_Public/UCSF_BrainMetastases_v1.3
 * - targetDatasetRoot:
 *   /mnt/d/A1_RainSun_20240916/1-UWMadison/IDiA-Lab/brain_metastases_train/nnUnet_dataset/nnUnet_raw/Dataset001_UCSFbrainmets
 *
 * The program also writes renaming_map.csv in targetDatasetRoot:
 *   SubjectID,nnUNet_case_id,FLAIR,T1pre,T1post,T2Synth,Label
 */
public class reformat {

    // Default raw UCSF source folder (can be overridden via args[0]).
    private static final String DEFAULT_SOURCE_ROOT =
            "/mnt/d/A1_RainSun_20240916/1-UWMadison/IDiA-Lab/Medical_Images_Public/UCSF_BrainMetastases_v1.3";
    // Default nnU-Net dataset target root (can be overridden via args[1]).
    private static final String DEFAULT_TARGET_DATASET_ROOT =
            "/mnt/d/A1_RainSun_20240916/1-UWMadison/IDiA-Lab/brain_metastases_train/nnUnet_dataset/nnUnet_raw/Dataset001_UCSFbrainmets";

    // Prefix used for generated nnU-Net case IDs.
    private static final String CASE_PREFIX = "UCSFbrainmets";

    // Maps source filename suffixes to nnU-Net channel IDs.
    private enum Modality {
        FLAIR("FLAIR", "0000"),
        T1PRE("T1pre", "0001"),
        T1POST("T1post", "0002"),
        T2SYNTH("T2Synth", "0003");

        final String sourceSuffix;
        final String nnUnetCode;

        Modality(String sourceSuffix, String nnUnetCode) {
            this.sourceSuffix = sourceSuffix;
            this.nnUnetCode = nnUnetCode;
        }
    }

    private static final String LABEL_SUFFIX = "BraTS-seg";

    // Holds discovered paths for one SubjectID.
    private static class CaseFiles {
        final Map<Modality, Path> modalities = new EnumMap<>(Modality.class);
        Path label;
    }

    public static void main(String[] args) throws IOException {
        // Resolve source/target roots from args or defaults.
        Path sourceRoot = args.length > 0 ? Paths.get(args[0]) : Paths.get(DEFAULT_SOURCE_ROOT);
        Path targetDatasetRoot = args.length > 1 ? Paths.get(args[1]) : Paths.get(DEFAULT_TARGET_DATASET_ROOT);
        Path imagesTr = targetDatasetRoot.resolve("imagesTr");
        Path labelsTr = targetDatasetRoot.resolve("labelsTr");

        // Fail fast if source root is invalid.
        if (!Files.isDirectory(sourceRoot)) {
            throw new IOException("Source root does not exist or is not a directory: " + sourceRoot);
        }

        // Ensure nnU-Net required folders exist.
        Files.createDirectories(imagesTr);
        Files.createDirectories(labelsTr);
        System.out.println("Starting UCSF -> nnU-Net reformat");
        System.out.println("Source root: " + sourceRoot);
        System.out.println("Target dataset root: " + targetDatasetRoot);
        System.out.println("Target imagesTr: " + imagesTr);
        System.out.println("Target labelsTr: " + labelsTr);
        System.out.println("Scanning source files...");

        // Discover all candidate cases by parsing source filenames.
        Map<String, CaseFiles> discovered = discoverCases(sourceRoot);
        List<String> subjects = new ArrayList<>(discovered.keySet());
        Collections.sort(subjects);
        System.out.println("Discovered candidate SubjectIDs: " + subjects.size());
        System.out.println("Beginning copy/organize phase...");

        int copiedCases = 0;
        int skippedCases = 0;
        int skippedMissingLabel = 0;
        List<String> skipped = new ArrayList<>();
        List<String> skippedNoLabel = new ArrayList<>();
        List<String[]> mappingRows = new ArrayList<>();

        for (String subjectId : subjects) {
            CaseFiles cf = discovered.get(subjectId);
            // Explicitly exclude cases without BraTS segmentation labels.
            if (cf == null || cf.label == null) {
                skippedCases++;
                skippedMissingLabel++;
                skippedNoLabel.add(subjectId);
                System.out.println("[SKIP][NO_LABEL] SubjectID=" + subjectId + " (missing _BraTS-seg.nii.gz)");
                continue;
            }
            if (!isCompleteCase(cf)) {
                skippedCases++;
                skipped.add(subjectId);
                System.out.println("[SKIP][INCOMPLETE] SubjectID=" + subjectId + " (missing modality file)");
                continue;
            }

            // Assign contiguous nnU-Net case IDs only to included cases.
            String nnCaseId = String.format(Locale.ROOT, "%s_%03d", CASE_PREFIX, copiedCases);
            System.out.println("[COPY] SubjectID=" + subjectId + " -> " + nnCaseId);

            // Copy all modality channels to imagesTr with nnU-Net channel suffixes.
            for (Modality m : Modality.values()) {
                Path src = cf.modalities.get(m);
                Path dst = imagesTr.resolve(nnCaseId + "_" + m.nnUnetCode + ".nii.gz");
                copyFile(src, dst);
                System.out.println("  [IMG] " + m.sourceSuffix + " -> " + dst.getFileName());
            }
            // Copy segmentation label to labelsTr with case-only filename.
            Path dstLabel = labelsTr.resolve(nnCaseId + ".nii.gz");
            copyFile(cf.label, dstLabel);
            System.out.println("  [LBL] BraTS-seg -> " + dstLabel.getFileName());

            // Record traceability row: original SubjectID -> generated nnU-Net case ID.
            mappingRows.add(new String[] {
                    subjectId,
                    nnCaseId,
                    cf.modalities.get(Modality.FLAIR).toString(),
                    cf.modalities.get(Modality.T1PRE).toString(),
                    cf.modalities.get(Modality.T1POST).toString(),
                    cf.modalities.get(Modality.T2SYNTH).toString(),
                    cf.label.toString()
            });

            copiedCases++;
            System.out.println("  [DONE] " + nnCaseId);
        }

        Path mappingCsv = targetDatasetRoot.resolve("renaming_map.csv");
        writeMappingCsv(mappingCsv, mappingRows);
        // Generate minimal nnU-Net v2 dataset.json metadata.
        writeDatasetJson(targetDatasetRoot.resolve("dataset.json"), copiedCases);

        System.out.println("Done.");
        System.out.println("Source root: " + sourceRoot);
        System.out.println("Target dataset root: " + targetDatasetRoot);
        System.out.println("Copied complete cases: " + copiedCases);
        System.out.println("Skipped incomplete cases: " + skippedCases);
        System.out.println("Skipped due to missing BraTS-seg label: " + skippedMissingLabel);
        if (!skipped.isEmpty()) {
            System.out.println("Skipped SubjectIDs (missing modality/label):");
            for (String sid : skipped) {
                System.out.println("  - " + sid);
            }
        }
        if (!skippedNoLabel.isEmpty()) {
            System.out.println("Skipped SubjectIDs (missing BraTS-seg):");
            for (String sid : skippedNoLabel) {
                System.out.println("  - " + sid);
            }
        }
        System.out.println("Mapping file: " + mappingCsv);
        System.out.println("dataset.json: " + targetDatasetRoot.resolve("dataset.json"));
    }

    private static Map<String, CaseFiles> discoverCases(Path sourceRoot) throws IOException {
        // SubjectID -> discovered modality/label files.
        Map<String, CaseFiles> bySubject = new LinkedHashMap<>();

        FileVisitor<Path> visitor = new SimpleFileVisitor<>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) {
                String name = file.getFileName().toString();
                if (!name.endsWith(".nii.gz")) {
                    return FileVisitResult.CONTINUE;
                }

                for (Modality m : Modality.values()) {
                    String suffix = "_" + m.sourceSuffix + ".nii.gz";
                    if (name.endsWith(suffix)) {
                        // Source modality filename format: {SubjectID}_{Modality}.nii.gz
                        String subjectId = name.substring(0, name.length() - suffix.length());
                        CaseFiles cf = bySubject.computeIfAbsent(subjectId, k -> new CaseFiles());
                        cf.modalities.put(m, file);
                        return FileVisitResult.CONTINUE;
                    }
                }

                String labelSuffix = "_" + LABEL_SUFFIX + ".nii.gz";
                if (name.endsWith(labelSuffix)) {
                    // Source label filename format: {SubjectID}_BraTS-seg.nii.gz
                    String subjectId = name.substring(0, name.length() - labelSuffix.length());
                    CaseFiles cf = bySubject.computeIfAbsent(subjectId, k -> new CaseFiles());
                    cf.label = file;
                }
                return FileVisitResult.CONTINUE;
            }
        };

        Files.walkFileTree(sourceRoot, visitor);
        return bySubject;
    }

    private static boolean isCompleteCase(CaseFiles cf) {
        // A valid case requires one label and all configured modalities.
        if (cf == null || cf.label == null) {
            return false;
        }
        for (Modality m : Modality.values()) {
            if (!cf.modalities.containsKey(m)) {
                return false;
            }
        }
        return true;
    }

    private static void copyFile(Path src, Path dst) throws IOException {
        // Overwrite existing target files to support reruns.
        Files.createDirectories(dst.getParent());
        Files.copy(src, dst, StandardCopyOption.REPLACE_EXISTING);
    }

    private static void writeMappingCsv(Path outputCsv, List<String[]> rows) throws IOException {
        // CSV provides auditable mapping from original file IDs to new nnU-Net IDs.
        try (BufferedWriter writer = Files.newBufferedWriter(outputCsv)) {
            writer.write("SubjectID,nnUNet_case_id,FLAIR,T1pre,T1post,T2Synth,Label");
            writer.newLine();
            for (String[] row : rows) {
                writer.write(csv(row[0]) + "," + csv(row[1]) + "," + csv(row[2]) + "," + csv(row[3]) + ","
                        + csv(row[4]) + "," + csv(row[5]) + "," + csv(row[6]));
                writer.newLine();
            }
        }
    }

    private static String csv(String s) {
        if (s == null) {
            return "";
        }
        String escaped = s.replace("\"", "\"\"");
        return "\"" + escaped + "\"";
    }

    private static void writeDatasetJson(Path datasetJson, int numTraining) throws IOException {
        // Minimal nnU-Net v2 dataset metadata for this 4-channel configuration.
        String json = "{\n"
                + "  \"channel_names\": {\n"
                + "    \"0\": \"FLAIR\",\n"
                + "    \"1\": \"T1pre\",\n"
                + "    \"2\": \"T1post\",\n"
                + "    \"3\": \"T2Synth\"\n"
                + "  },\n"
                + "  \"labels\": {\n"
                + "    \"background\": 0,\n"
                + "    \"label_1\": 1,\n"
                + "    \"label_2\": 2,\n"
                + "    \"label_3\": 3\n"
                + "  },\n"
                + "  \"numTraining\": " + numTraining + ",\n"
                + "  \"file_ending\": \".nii.gz\"\n"
                + "}\n";
        Files.writeString(datasetJson, json);
    }
}