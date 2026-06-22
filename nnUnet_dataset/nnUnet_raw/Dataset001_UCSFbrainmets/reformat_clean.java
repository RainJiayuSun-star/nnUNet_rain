import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Removes T2/T2Synth channel files (_0003.nii.gz) from nnU-Net imagesTr.
 *
 * Also rewrites dataset.json for 3-channel usage:
 * - 0000 FLAIR
 * - 0001 T1pre
 * - 0002 T1post
 *
 * Usage:
 *   java reformat_clean [datasetRoot]
 *
 * Default dataset root:
 * /mnt/d/A1_RainSun_20240916/1-UWMadison/IDiA-Lab/brain_metastases_train/nnUnet_dataset/nnUnet_raw/Dataset001_UCSFbrainmets
 */
public class reformat_clean {

    // Default dataset folder; override with CLI arg if needed.
    private static final String DEFAULT_DATASET_ROOT =
            "/mnt/d/A1_RainSun_20240916/1-UWMadison/IDiA-Lab/brain_metastases_train/nnUnet_dataset/nnUnet_raw/Dataset001_UCSFbrainmets";

    // Matches nnU-Net channel files for modality index 0003 (T2/T2Synth).
    private static final Pattern T2_PATTERN = Pattern.compile("^.+_0003\\.nii\\.gz$");

    public static void main(String[] args) throws IOException {
        // Resolve dataset root from argument or fallback to default.
        Path datasetRoot = args.length > 0 ? Paths.get(args[0]) : Paths.get(DEFAULT_DATASET_ROOT);
        Path imagesTr = datasetRoot.resolve("imagesTr");
        Path labelsTr = datasetRoot.resolve("labelsTr");
        Path datasetJson = datasetRoot.resolve("dataset.json");

        // Validate required directories before mutation.
        if (!Files.isDirectory(datasetRoot)) {
            throw new IOException("Dataset root does not exist: " + datasetRoot);
        }
        if (!Files.isDirectory(imagesTr)) {
            throw new IOException("imagesTr does not exist: " + imagesTr);
        }

        System.out.println("Cleaning nnU-Net dataset to 3 channels");
        System.out.println("Dataset root: " + datasetRoot);
        System.out.println("imagesTr: " + imagesTr);
        System.out.println("labelsTr: " + labelsTr);

        List<Path> deleted = new ArrayList<>();
        int keptFiles = 0;

        // Scan imagesTr and delete only the 0003 modality files.
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(imagesTr)) {
            for (Path p : stream) {
                if (!Files.isRegularFile(p)) {
                    continue;
                }
                String name = p.getFileName().toString();
                if (T2_PATTERN.matcher(name).matches()) {
                    Files.deleteIfExists(p);
                    deleted.add(p);
                    System.out.println("[DELETE] " + name);
                } else {
                    keptFiles++;
                }
            }
        }

        // Recount files after cleanup for consistency reporting.
        int labelCount = countRegularFiles(labelsTr);
        int imageCount = countRegularFiles(imagesTr);

        // Rewrite dataset.json to reflect 3-channel training.
        String json = "{\n"
                + "  \"channel_names\": {\n"
                + "    \"0\": \"FLAIR\",\n"
                + "    \"1\": \"T1pre\",\n"
                + "    \"2\": \"T1post\"\n"
                + "  },\n"
                + "  \"labels\": {\n"
                + "    \"background\": 0,\n"
                + "    \"label_1\": 1,\n"
                + "    \"label_2\": 2,\n"
                + "    \"label_3\": 3\n"
                + "  },\n"
                + "  \"numTraining\": " + labelCount + ",\n"
                + "  \"file_ending\": \".nii.gz\"\n"
                + "}\n";
        Files.writeString(datasetJson, json);

        System.out.println("Done.");
        System.out.println("Deleted _0003 files: " + deleted.size());
        System.out.println("Kept non-_0003 files seen during scan: " + keptFiles);
        System.out.println("Current imagesTr file count: " + imageCount);
        System.out.println("Current labelsTr file count: " + labelCount);
        System.out.println("Updated dataset.json: " + datasetJson);

        // For a complete 3-channel dataset, image count should be exactly 3x labels.
        if (labelCount > 0 && imageCount != labelCount * 3) {
            System.out.println("[WARN] imagesTr count is not 3x labelsTr. Please verify dataset consistency.");
        }
    }

    private static int countRegularFiles(Path dir) throws IOException {
        // Utility: count only regular files directly inside the directory.
        if (!Files.isDirectory(dir)) {
            return 0;
        }
        int count = 0;
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
            for (Path p : stream) {
                if (Files.isRegularFile(p)) {
                    count++;
                }
            }
        }
        return count;
    }
}
