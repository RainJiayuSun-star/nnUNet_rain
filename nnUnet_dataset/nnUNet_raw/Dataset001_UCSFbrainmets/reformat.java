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
 * Output naming (we preserve only FLAIR and T1post):
 * - imagesTr/UCSFbrainmets_[xxx]_0000.nii.gz (FLAIR)
 * - imagesTr/UCSFbrainmets_[xxx]_0001.nii.gz (T1post)
 * - labelsTr/UCSFbrainmets_[xxx].nii.gz      (BraTS-seg label)
 *
 * Usage:
 *   java reformat [sourceRoot] [targetDatasetRoot]
 *
 * Defaults:
 * - sourceRoot:
 *   /mnt/local/data/rainsun/metastases/datasets_preprocessed/UCSF_BrainMetastases_TRAIN
 * - targetDatasetRoot:
 *   /mnt/local/data/rainsun/metastases/Rain-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset/nnUnet_raw/Dataset001_UCSFbrainmets
 *
 * The program also writes renaming_map.csv in targetDatasetRoot:
 *   SubjectID,nnUNet_case_id,FLAIR,T1post,Label
 */
public class reformat {

    // Default raw UCSF source folder (can be overridden via args[0]).
        private static final String DEFAULT_SOURCE_ROOT =
            "/app/datasets_preprocessed/UCSF_BrainMetastases_TRAIN";
        // Default nnU-Net dataset target root (can be overridden via args[1]).
        private static final String DEFAULT_TARGET_DATASET_ROOT =
            "/app/IDIA-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset/nnUNet_raw/Dataset001_UCSFbrainmets";

    // Prefix used for generated nnU-Net case IDs.
    private static final String CASE_PREFIX = "UCSFbrainmets";

    // Maps source filename suffixes (case-insensitive) to nnUnet channel IDs.
    private enum Modality {
        FLAIR(new String[]{"flair", "flair_bc"}, "0000"),
        // Accept both t1post and t1ce as the post-contrast structural scan; include _BC variants
        T1POST(new String[]{"t1post", "t1post_bc", "t1ce", "t1ce_bc"}, "0001");

        final String[] sourceSuffixes; // lower-case suffixes
        final String nnUnetCode;

        Modality(String[] sourceSuffixes, String nnUnetCode) {
            this.sourceSuffixes = sourceSuffixes;
            this.nnUnetCode = nnUnetCode;
        }
        String displayName() { return sourceSuffixes[0]; }
    }

    // Only accept the requested segmentation file name format:
    // {SubjectID}_combined.seg.nii.gz
    private static final String LABEL_SUFFIX = "_combined.seg.nii.gz";

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

        int debugPrinted = 0;
        for (String subjectId : subjects) {
            CaseFiles cf = discovered.get(subjectId);
            if (debugPrinted < 10) {
                System.out.println("[DBG] SubjectID=" + subjectId + " modalities=" + cf.modalities.keySet() + " label=" + (cf.label != null));
                debugPrinted++;
            }
            // Explicitly exclude cases without BraTS segmentation labels.
            if (cf == null || cf.label == null) {
                skippedCases++;
                skippedMissingLabel++;
                skippedNoLabel.add(subjectId);
                System.out.println("[SKIP][NO_LABEL] SubjectID=" + subjectId + " (missing _combined.seg.nii.gz)");
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
                System.out.println("  [IMG] " + m.displayName() + " -> " + dst.getFileName());
            }
            // Copy segmentation label to labelsTr with case-only filename.
            Path dstLabel = labelsTr.resolve(nnCaseId + ".nii.gz");
            copyFile(cf.label, dstLabel);
            System.out.println("  [LBL] combined.seg -> " + dstLabel.getFileName());

                // Record traceability row: original SubjectID -> generated nnU-Net case ID.
                mappingRows.add(new String[] {
                    subjectId,
                    nnCaseId,
                    cf.modalities.get(Modality.FLAIR).toString(),
                    cf.modalities.get(Modality.T1POST).toString(),
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
        System.out.println("Skipped due to missing combined.seg label: " + skippedMissingLabel);
        if (!skipped.isEmpty()) {
            System.out.println("Skipped SubjectIDs (missing modality/label):");
            for (String sid : skipped) {
                System.out.println("  - " + sid);
            }
        }
        if (!skippedNoLabel.isEmpty()) {
            System.out.println("Skipped SubjectIDs (missing combined.seg):");
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

                String nameLower = name.toLowerCase(Locale.ROOT);
                for (Modality m : Modality.values()) {
                    for (String sfx : m.sourceSuffixes) {
                        String suffix = "_" + sfx + ".nii.gz";
                        if (nameLower.endsWith(suffix)) {
                            // Source modality filename format: {SubjectID}_{Modality}.nii.gz
                            String subjectId = name.substring(0, name.length() - suffix.length());
                            CaseFiles cf = bySubject.computeIfAbsent(subjectId, k -> new CaseFiles());
                            cf.modalities.put(m, file);
                            return FileVisitResult.CONTINUE;
                        }
                    }
                }

                if (nameLower.endsWith(LABEL_SUFFIX)) {
                    // Source label filename format: {SubjectID}_combined.seg.nii.gz
                    String subjectId = name.substring(0, name.length() - LABEL_SUFFIX.length());
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
            writer.write("SubjectID,nnUNet_case_id,FLAIR,T1post,Label");
            writer.newLine();
            for (String[] row : rows) {
                writer.write(csv(row[0]) + "," + csv(row[1]) + "," + csv(row[2]) + "," + csv(row[3]) + ","
                        + csv(row[4]));
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
        // Minimal nnU-Net v2 dataset metadata for the 2-channel configuration (FLAIR, T1post).
        String json = "{\n"
            + "  \"channel_names\": {\n"
            + "    \"0\": \"FLAIR\",\n"
            + "    \"1\": \"T1post\"\n"
            + "  },\n"
            + "  \"labels\": {\n"
            + "    \"background\": 0,\n"
            + "    \"tumor_core\": 1,\n"
            + "    \"edema\": 2\n"
            + "  },\n"
            + "  \"numTraining\": " + numTraining + ",\n"
            + "  \"file_ending\": \".nii.gz\"\n"
            + "}\n";
        Files.writeString(datasetJson, json);
    }
}