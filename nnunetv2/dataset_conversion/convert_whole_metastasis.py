"""Convert verified combined core/edema labels to binary whole metastasis.

Run with python -m nnunetv2.dataset_conversion.convert_whole_metastasis --help.
This accepts COMBINED labels, never raw BraTS label IDs. All input validation
finishes before destination creation. Existing destinations are never replaced.
"""
import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_id(path):
    match = re.fullmatch(r"Dataset([0-9]{3})_.+", path.name)
    if not match:
        raise ValueError(f"Expected DatasetNNN_Name directory: {path}")
    return match[1]


def read_volume(path):
    image = sitk.ReadImage(str(path))
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError(f"Expected scalar 3D NIfTI: {path}")
    return image, sitk.GetArrayFromImage(image)


def same_geometry(a, b):
    return (a.GetSize() == b.GetSize()
            and np.allclose(a.GetSpacing(), b.GetSpacing(), rtol=0, atol=1e-5)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), rtol=0, atol=1e-5)
            and np.allclose(a.GetDirection(), b.GetDirection(), rtol=0, atol=1e-5))


def validate_splits(splits, patients):
    cases = set(patients)
    if not isinstance(splits, list) or len(splits) < 2:
        raise ValueError("Expected cross-validation split list with at least two folds")
    validation_cases = []
    for fold in splits:
        if not isinstance(fold, dict) or set(fold) != {"train", "val"}:
            raise ValueError("Each fold must contain train and val lists")
        train, val = fold["train"], fold["val"]
        if not isinstance(train, list) or not isinstance(val, list) or not train or not val:
            raise ValueError("Empty or invalid train/val lists")
        if len(set(train)) != len(train) or len(set(val)) != len(val):
            raise ValueError("Duplicate split case")
        if set(train) & set(val) or set(train) | set(val) != cases:
            raise ValueError("Split case identities differ from the dataset or overlap")
        if {patients[c] for c in train} & {patients[c] for c in val}:
            raise ValueError("Patient leakage between training and validation")
        validation_cases.extend(val)
    if len(validation_cases) != len(cases) or set(validation_cases) != cases:
        raise ValueError("Every case must occur in validation exactly once across folds")


def convert(source, destination, *, mapping=None, splits=None, audit_dir=None, dry_run=False):
    if Path(destination).is_symlink():
        raise FileExistsError("Destination is a symlink; use a fresh directory")
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if dataset_id(source) == dataset_id(destination):
        raise ValueError("Destination must have a distinct dataset ID")
    if source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination cannot be nested")
    if destination.exists():
        raise FileExistsError(f"Destination already exists; use a fresh directory: {destination}")
    if not destination.parent.is_dir():
        raise ValueError("Destination parent must already exist")
    # Catch a different name using the same ID in the target raw-data root.
    if list(destination.parent.glob(f"Dataset{dataset_id(destination)}_*")):
        raise FileExistsError("Destination dataset ID is already used in this raw-data root")
    metadata = json.loads((source / "dataset.json").read_text())
    if metadata.get("labels") != {"background": 0, "tumor_core": 1, "edema": 2}:
        raise ValueError("Source must declare combined labels background=0, tumor_core=1, edema=2")
    if metadata.get("file_ending") != ".nii.gz" or "regions_class_order" in metadata:
        raise ValueError("Source must be label-based .nii.gz data")
    channels = metadata.get("channel_names", {})
    if channels != {"0": "FLAIR", "1": "T1post"}:
        raise ValueError("Expected Dataset003 channels FLAIR=0, T1post=1")
    labels = sorted((source / "labelsTr").glob("*.nii.gz"))
    if not labels or len(labels) != metadata.get("numTraining"):
        raise ValueError("numTraining does not match labelsTr")
    cases = {p.name[:-7] for p in labels}
    mapping = Path(mapping).resolve() if mapping else source / "renaming_map.csv"
    with mapping.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if (len(rows) != len(cases) or {r.get("nnUNet_case_id") for r in rows} != cases
            or any(not r.get("patient_id", "").strip() for r in rows)):
        raise ValueError("Mapping must have exactly one nnUNet_case_id and patient_id per case")
    patients = {r["nnUNet_case_id"]: r["patient_id"].strip() for r in rows}
    subjects = {r["nnUNet_case_id"]: r.get("SubjectID", "") for r in rows}
    split_data = json.loads(Path(splits).read_text()) if splits else None
    if split_data is not None:
        validate_splits(split_data, patients)
    elif len(set(patients.values())) != len(patients):
        raise ValueError("Multiple scans per patient require explicit patient-disjoint splits")
    expected_images = {f"{case}_{int(c):04d}.nii.gz" for case in cases for c in channels}
    if {p.name for p in (source / "imagesTr").glob("*.nii.gz")} != expected_images:
        raise ValueError("imagesTr does not match cases and channels")
    audit_dir = Path(audit_dir).resolve() if audit_dir else None
    if audit_dir is not None and not audit_dir.is_dir():
        raise ValueError("Audit directory does not exist")
    manifest = {
        "schema_version": 1, "source": str(source), "destination": str(destination),
        "foreground": "enhancing plus non-enhancing/necrotic metastasis, excluding edema",
        "mapping": {"0": 0, "1": 1, "2": 0},
        "source_dataset_sha256": sha256(source / "dataset.json"),
        "patient_mapping_sha256": sha256(mapping),
        "splits": "validated and supplied" if splits else "not supplied; use identical generated folds for all trainers",
        "source_splits_sha256": sha256(splits) if splits else None,
        "audit_definition": "same-space raw annotation IDs 1,3,4 union equals combined label 1",
        "cases": [],
    }
    # Preflight every case before writing any output; no resampling or registration.
    for path in labels:
        case = path.name[:-7]
        label_image, values = read_volume(path)
        if not np.isin(values, [0, 1, 2]).all():
            raise ValueError(f"Unexpected label values: {path}")
        record = {"case_id": case, "patient_id": patients[case],
                  "source_label": str(path), "source_label_sha256": sha256(path),
                  "source_label_counts": {str(v): int(np.count_nonzero(values == v)) for v in (0, 1, 2)},
                  "foreground_voxels": int(np.count_nonzero(values == 1)), "images": [],
                  "audit": "not_verified: original same-space annotation not supplied"}
        for channel in channels:
            image_path = source / "imagesTr" / f"{case}_{int(channel):04d}.nii.gz"
            image, array = read_volume(image_path)
            if not same_geometry(label_image, image) or not np.isfinite(array).all():
                raise ValueError(f"Image/label geometry mismatch or nonfinite image: {image_path}")
            record["images"].append({"path": str(image_path), "sha256": sha256(image_path)})
        if audit_dir is not None:
            original = audit_dir / path.name
            subject = subjects[case]
            # Support the original UCSF layout using the explicit provenance map.
            # Do not guess a standard *_seg annotation's label semantics.
            if not original.exists() and subject:
                if Path(subject).name != subject or subject in (".", ".."):
                    raise ValueError(f"Invalid SubjectID in mapping: {subject}")
                original = audit_dir / subject / f"{subject}_BraTS-seg.nii.gz"
            if original.is_file():
                original_image, original_values = read_volume(original)
                if not same_geometry(label_image, original_image):
                    raise ValueError(f"Audit annotation is not in the same image space: {original}")
                if not np.isin(original_values, [0, 1, 2, 3, 4]).all():
                    raise ValueError(f"Unknown raw annotation IDs: {original}")
                if not np.array_equal(values == 1, np.isin(original_values, [1, 3, 4])):
                    raise ValueError(f"Whole-metastasis annotation mismatch: {original}")
                record["audit"] = "verified_union"
                record["audit_path"] = str(original)
                record["audit_sha256"] = sha256(original)
            else:
                record["audit"] = "not_verified: case annotation absent from audit directory"
        manifest["cases"].append(record)
    manifest["audited_cases"] = sum(r["audit"] == "verified_union" for r in manifest["cases"])
    manifest["unaudited_cases"] = len(labels) - manifest["audited_cases"]
    if dry_run:
        return manifest
    destination.mkdir()  # Exclusive creation; do not overwrite another conversion.
    (destination / "imagesTr").mkdir()
    (destination / "labelsTr").mkdir()
    output_metadata = dict(metadata)
    output_metadata["labels"] = {"background": 0, "whole_metastasis": 1}
    output_metadata["name"] = destination.name
    (destination / "dataset.json").write_text(json.dumps(output_metadata, indent=2) + "\n")
    shutil.copy2(mapping, destination / "renaming_map.csv")
    if split_data is not None:
        (destination / "splits_final.json").write_text(json.dumps(split_data, indent=2) + "\n")
    for record in manifest["cases"]:
        path = Path(record["source_label"])
        if sha256(path) != record["source_label_sha256"]:
            raise ValueError(f"Source label changed after preflight: {path}")
        image, values = read_volume(path)
        binary = sitk.GetImageFromArray((values == 1).astype(np.uint8))
        binary.CopyInformation(image)
        output = destination / "labelsTr" / path.name
        sitk.WriteImage(binary, str(output))
        record["output_label_sha256"] = sha256(output)
        for item in record["images"]:
            image_path = Path(item["path"])
            copied = destination / "imagesTr" / image_path.name
            shutil.copy2(image_path, copied)
            if sha256(copied) != item["sha256"]:
                raise ValueError(f"Source image changed after preflight: {image_path}")
    (destination / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, help="Defaults to source/renaming_map.csv")
    parser.add_argument("--splits", type=Path, help="Existing source splits_final.json")
    parser.add_argument("--audit-dir", type=Path, help="Same-space originals: CASE_ID.nii.gz or SubjectID/SubjectID_BraTS-seg.nii.gz; tumor IDs 1/3/4")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = convert(args.source, args.destination, mapping=args.mapping, splits=args.splits,
                     audit_dir=args.audit_dir, dry_run=args.dry_run)
    print(json.dumps({"dry_run": args.dry_run, "cases": len(report["cases"]),
                      "audited_cases": report["audited_cases"], "unaudited_cases": report["unaudited_cases"],
                      "destination": report["destination"], "splits": report["splits"]}, indent=2))
    if report["unaudited_cases"]:
        print("WARNING: label values were checked, but anatomical provenance is NOT fully verified. Review before training.")


if __name__ == "__main__":
    main()
