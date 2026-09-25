import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.*;

/** Java 21: java reformat.java [sourceRoot] [targetDatasetRoot] [cohortCsv] [--dry-run]. */
public class reformat {
    private static final String DEFAULT_SOURCE_ROOT =
        "/app/datasets_preprocessed/UCSF_BrainMetastases_TRAIN";
    private static final String DEFAULT_TARGET_DATASET_ROOT =
        "/app/IDIA-BrainMetastases-main/train/nnUNet_rain/nnUnet_dataset/nnUNet_raw/Dataset003_UCSFbrainmets_222";
    private static final String COHORT_FILE = "ucsf_scan_A_preoperative_brats_training_222.csv";
    private static final int EXPECTED_CASES = 222;
    private static final List<String> MAP_HEADER = List.of("SubjectID", "nnUNet_case_id", "FLAIR", "T1post", "Label", "patient_id", "brats_id");
    private static final String DATASET_JSON = """
        {
          "channel_names": {"0": "FLAIR", "1": "T1post"},
          "labels": {"background": 0, "tumor_core": 1, "edema": 2},
          "numTraining": 222,
          "file_ending": ".nii.gz"
        }
        """;
    private record Patient(String id, String subject, String brats) {}
    private record Copy(Path source, Path target) {}

    public static void main(String[] args) {
        try { run(args); }
        catch (IOException | IllegalArgumentException e) {
            System.err.println("Reformat failed: " + e.getMessage());
            System.exit(1);
        }
    }

    private static void run(String[] args) throws IOException {
        List<String> positional = new ArrayList<>();
        boolean dry = false;
        for (String arg : args) {
            if (arg.equals("--dry-run")) dry = true;
            else if (arg.startsWith("--")) throw new IOException("Unknown option: " + arg);
            else positional.add(arg);
        }
        if (positional.size() > 3) throw new IOException("Usage: java reformat.java [sourceRoot] [targetDatasetRoot] [cohortCsv] [--dry-run]");
        Path source = Path.of(positional.size() > 0 ? positional.get(0) : DEFAULT_SOURCE_ROOT).toRealPath();
        Path target = Path.of(positional.size() > 1 ? positional.get(1) : DEFAULT_TARGET_DATASET_ROOT).toAbsolutePath().normalize();
        if (!Files.isDirectory(source)) throw new IOException("Not a source directory: " + source);
        Path resolvedTarget = resolveExistingParent(target);
        if (resolvedTarget.startsWith(source) || source.startsWith(resolvedTarget))
            throw new IOException("Source and target directories must not overlap.");
        Path cohort = positional.size() > 2 ? Path.of(positional.get(2)) : target.resolve(COHORT_FILE);
        List<Patient> patients = readCohort(cohort);
        Map<String, List<Path>[]> files = new TreeMap<>();
        for (Patient p : patients) files.put(p.subject(), newLists());
        // Recursively inspect filenames, but retain candidates only for allowlisted examinations.
        Files.walkFileTree(source, new SimpleFileVisitor<Path>() {
            @Override public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) {
                String name = file.getFileName().toString();
                int underscore = name.indexOf('_');
                if (underscore < 0) return FileVisitResult.CONTINUE;
                List<Path>[] matches = files.get(name.substring(0, underscore));
                if (matches == null) return FileVisitResult.CONTINUE;
                String suffix = name.substring(underscore).toLowerCase(Locale.ROOT);
                int channel = switch (suffix) {
                    case "_flair.nii.gz", "_flair_bc.nii.gz" -> 0;
                    case "_t1post.nii.gz", "_t1post_bc.nii.gz", "_t1ce.nii.gz", "_t1ce_bc.nii.gz" -> 1;
                    case "_combined.seg.nii.gz" -> 2;
                    default -> -1;
                };
                if (channel >= 0) matches[channel].add(file);
                return FileVisitResult.CONTINUE;
            }
        });
        List<String> errors = new ArrayList<>();
        String[] names = {"FLAIR", "T1post", "combined.seg"};
        for (Patient p : patients) {
            List<Path>[] matches = files.get(p.subject());
            for (int i = 0; i < 3; i++) {
                if (matches[i].size() != 1) errors.add(p.subject() + " " + names[i] + ": expected one file; found " + matches[i]);
                else if (!Files.isRegularFile(matches[i].get(0)) || Files.size(matches[i].get(0)) == 0)
                    errors.add(p.subject() + " " + names[i] + ": empty or not a regular file: " + matches[i].get(0));
            }
        }
        if (!errors.isEmpty()) throw new IOException("Source preflight failed before copying:\n" + String.join("\n", errors));
        List<List<String>> mapping = new ArrayList<>();
        mapping.add(MAP_HEADER);
        List<Copy> copies = new ArrayList<>();
        for (int i = 0; i < patients.size(); i++) {
            Patient p = patients.get(i);
            String caseId = String.format(Locale.ROOT, "UCSFbrainmets_%03d", i);
            List<Path>[] paths = files.get(p.subject());
            Path flair = paths[0].get(0), t1 = paths[1].get(0), label = paths[2].get(0);
            copies.add(new Copy(flair, target.resolve("imagesTr/" + caseId + "_0000.nii.gz")));
            copies.add(new Copy(t1, target.resolve("imagesTr/" + caseId + "_0001.nii.gz")));
            copies.add(new Copy(label, target.resolve("labelsTr/" + caseId + ".nii.gz")));
            mapping.add(List.of(p.subject(), caseId, flair.toString(), t1.toString(), label.toString(), p.id(), p.brats()));
        }
        validateTarget(target, copies, mapping);
        System.out.println("Source: " + source + "\nTarget: " + target + "\nCohort: " + cohort.toAbsolutePath());
        System.out.println("Validated 222 patients: 444 images, 222 combined labels; FLAIR=0000, T1post=0001.");
        if (dry) { System.out.println("Dry run complete; no files written."); return; }
        Files.createDirectories(target.resolve("imagesTr"));
        Files.createDirectories(target.resolve("labelsTr"));
        // Save the validated identity mapping first, so interrupted copies can be resumed safely.
        writeAtomic(target.resolve("renaming_map.csv"), toCsv(mapping));
        for (Copy copy : copies) {
            if (!Files.exists(copy.target())) {
                Path tmp = Files.createTempFile(target, ".reformat-copy-", ".tmp");
                try {
                    Files.copy(copy.source(), tmp, StandardCopyOption.REPLACE_EXISTING);
                    Files.move(tmp, copy.target(), StandardCopyOption.ATOMIC_MOVE);
                } finally { Files.deleteIfExists(tmp); }
            }
        }
        writeAtomic(target.resolve("dataset.json"), DATASET_JSON);
        validateTarget(target, copies, mapping);
        for (Copy copy : copies) if (!Files.isRegularFile(copy.target())) throw new IOException("Missing output: " + copy.target());
        System.out.println("Complete: 222 cases; renaming_map.csv and dataset.json written. No cases skipped.");
    }

    private static Path resolveExistingParent(Path path) throws IOException {
        if (Files.exists(path)) return path.toRealPath();
        return resolveExistingParent(path.getParent()).resolve(path.getFileName());
    }

    @SuppressWarnings("unchecked")
    private static List<Path>[] newLists() {
        return (List<Path>[]) new List<?>[]{new ArrayList<Path>(), new ArrayList<Path>(), new ArrayList<Path>()};
    }

    private static List<Patient> readCohort(Path path) throws IOException {
        List<List<String>> rows = readCsv(path);
        List<String> header = rows.get(0);
        for (String field : List.of("patient_id", "subject_id", "brats_id", "scan_letter", "prior_craniotomy_biopsy_resection"))
            if (!header.contains(field)) throw new IOException("Cohort missing column: " + field);
        List<Patient> patients = new ArrayList<>();
        Set<String> ids = new HashSet<>(), subjects = new HashSet<>(), bratsIds = new HashSet<>();
        for (List<String> row : rows.subList(1, rows.size())) {
            String id = row.get(header.indexOf("patient_id")), subject = row.get(header.indexOf("subject_id"));
            String brats = row.get(header.indexOf("brats_id"));
            if (!id.matches("100[0-9]{3}") || !subject.equals(id + "A") || id.equals("100414")
                || !row.get(header.indexOf("scan_letter")).equals("A")
                || !row.get(header.indexOf("prior_craniotomy_biopsy_resection")).equals("No")
                || !brats.matches("BraTS-MET-[0-9]{5}-000"))
                throw new IOException("Ineligible cohort row: " + row);
            if (!ids.add(id) || !subjects.add(subject) || !bratsIds.add(brats)) throw new IOException("Duplicate patient, examination or BraTS ID: " + row);
            patients.add(new Patient(id, subject, brats));
        }
        if (patients.size() != EXPECTED_CASES) throw new IOException("Expected 222 cohort rows; found " + patients.size());
        patients.sort(Comparator.comparing(Patient::subject));
        return patients;
    }

    private static void validateTarget(Path target, List<Copy> copies, List<List<String>> mapping) throws IOException {
        Set<Path> expected = new HashSet<>();
        for (Copy copy : copies) expected.add(copy.target());
        boolean hasOutput = false;
        for (String name : List.of("imagesTr", "labelsTr")) {
            Path dir = target.resolve(name);
            if (Files.isSymbolicLink(dir)) throw new IOException("Output directory must not be a symlink: " + dir);
            if (Files.exists(dir)) {
                try (var entries = Files.list(dir)) {
                    for (Path p : entries.toList()) {
                        hasOutput = true;
                        if (!expected.contains(p) || !Files.isRegularFile(p) || Files.isSymbolicLink(p))
                            throw new IOException("Unexpected output entry; nothing deleted: " + p);
                    }
                }
            }
        }
        Path map = target.resolve("renaming_map.csv"), json = target.resolve("dataset.json");
        if (Files.isSymbolicLink(map) || Files.isSymbolicLink(json)) throw new IOException("Generated metadata must not be symlinks.");
        if (Files.exists(map)) {
            if (!readCsv(map).equals(mapping)) throw new IOException("Existing renaming_map.csv conflicts with the planned cohort or source paths.");
        } else if (hasOutput || Files.exists(json)) throw new IOException("Existing outputs have no identity mapping; use a clean target.");
        if (Files.exists(json) && !Files.readString(json).replaceAll("\\s", "").equals(DATASET_JSON.replaceAll("\\s", "")))
            throw new IOException("Existing dataset.json conflicts with this formatter's metadata.");
        for (Copy copy : copies) if (Files.exists(copy.target()) && Files.mismatch(copy.source(), copy.target()) != -1)
            throw new IOException("Existing output differs from source; refusing overwrite: " + copy.target());
    }

    // Strict CSV reader: quoted fields, escaped quotes, embedded newlines, CRLF and UTF-8 BOM.
    private static List<List<String>> readCsv(Path path) throws IOException {
        String text = Files.readString(path);
        if (text.startsWith("\uFEFF")) text = text.substring(1);
        List<List<String>> rows = new ArrayList<>();
        List<String> row = new ArrayList<>();
        StringBuilder field = new StringBuilder();
        boolean quoted = false, closed = false;
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (quoted) {
                if (c == '"') {
                    if (i + 1 < text.length() && text.charAt(i + 1) == '"') { field.append('"'); i++; }
                    else { quoted = false; closed = true; }
                } else field.append(c);
            } else if (c == ',' || c == '\n' || c == '\r') {
                row.add(field.toString()); field.setLength(0); closed = false;
                if (c != ',') {
                    if (c == '\r' && i + 1 < text.length() && text.charAt(i + 1) == '\n') i++;
                    rows.add(row); row = new ArrayList<>();
                }
            } else if (c == '"' && field.length() == 0 && !closed) quoted = true;
            else {
                if (closed || c == '"') throw new IOException("Malformed CSV quoting: " + path);
                field.append(c);
            }
        }
        if (quoted) throw new IOException("Unterminated CSV quote: " + path);
        if (field.length() > 0 || closed || !row.isEmpty()) { row.add(field.toString()); rows.add(row); }
        if (rows.isEmpty()) throw new IOException("Empty CSV: " + path);
        if (new HashSet<>(rows.get(0)).size() != rows.get(0).size()) throw new IOException("Duplicate CSV headers: " + path);
        for (List<String> r : rows) if (r.size() != rows.get(0).size()) throw new IOException("Wrong CSV field count: " + path);
        return rows;
    }

    private static String toCsv(List<List<String>> rows) {
        StringBuilder text = new StringBuilder();
        for (List<String> row : rows) {
            List<String> fields = new ArrayList<>();
            for (String field : row) fields.add("\"" + field.replace("\"", "\"\"") + "\"");
            text.append(String.join(",", fields)).append('\n');
        }
        return text.toString();
    }

    private static void writeAtomic(Path target, String text) throws IOException {
        Path tmp = Files.createTempFile(target.getParent(), ".reformat-metadata-", ".tmp");
        try {
            Files.writeString(tmp, text);
            Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } finally { Files.deleteIfExists(tmp); }
    }
}
