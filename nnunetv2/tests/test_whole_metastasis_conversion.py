import csv
import json

import numpy as np
import pytest
import SimpleITK as sitk

from nnunetv2.dataset_conversion.convert_whole_metastasis import (
    convert,
    same_geometry,
    sha256,
    validate_splits,
)


def write_image(path, data, origin=(10, -20, 30)):
    image = sitk.GetImageFromArray(data)
    image.SetSpacing((0.7, 0.8, 1.5))
    image.SetOrigin(origin)
    image.SetDirection((-1, 0, 0, 0, -1, 0, 0, 0, 1))
    sitk.WriteImage(image, str(path))


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "Dataset003_Combined"
    (root / "labelsTr").mkdir(parents=True)
    (root / "imagesTr").mkdir()
    (root / "dataset.json").write_text(json.dumps({
        "labels": {"background": 0, "tumor_core": 1, "edema": 2},
        "channel_names": {"0": "FLAIR", "1": "T1post"}, "numTraining": 2, "file_ending": ".nii.gz"}))
    with (root / "renaming_map.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["nnUNet_case_id", "patient_id", "SubjectID"])
        writer.writeheader()
        for i in range(2):
            writer.writerow({"nnUNet_case_id": f"case{i}", "patient_id": f"patient{i}", "SubjectID": f"patient{i}A"})
    for i in range(2):
        data = np.zeros((4, 5, 6), dtype=np.uint8)
        data[1, 1, 1:3] = 1
        data[2, 1, 1:3] = 2
        write_image(root / "labelsTr" / f"case{i}.nii.gz", data)
        for c in range(2):
            write_image(root / "imagesTr" / f"case{i}_{c:04d}.nii.gz", data.astype(np.float32))
    return root


def test_convert_dry_run_geometry_manifest_and_splits(source, tmp_path):
    dest = tmp_path / "Dataset004_Binary"
    splits = tmp_path / "splits.json"
    folds = [{"train": ["case0"], "val": ["case1"]}, {"train": ["case1"], "val": ["case0"]}]
    splits.write_text(json.dumps(folds))
    before = sha256(source / "labelsTr/case0.nii.gz")
    report = convert(source, dest, splits=splits, dry_run=True)
    assert not dest.exists() and report["unaudited_cases"] == 2
    report = convert(source, dest, splits=splits)
    assert sha256(source / "labelsTr/case0.nii.gz") == before
    original = sitk.ReadImage(str(source / "labelsTr/case0.nii.gz"))
    binary = sitk.ReadImage(str(dest / "labelsTr/case0.nii.gz"))
    assert same_geometry(original, binary)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(binary), sitk.GetArrayFromImage(original) == 1)
    assert json.loads((dest / "dataset.json").read_text())["labels"] == {"background": 0, "whole_metastasis": 1}
    assert json.loads((dest / "splits_final.json").read_text()) == folds
    assert json.loads((dest / "conversion_manifest.json").read_text()) == report
    assert sha256(source / "renaming_map.csv") == sha256(dest / "renaming_map.csv")
    assert sha256(source / "imagesTr/case0_0000.nii.gz") == sha256(dest / "imagesTr/case0_0000.nii.gz")
    with pytest.raises(FileExistsError):
        convert(source, dest)


@pytest.mark.parametrize("bad", ["label", "geometry", "channels", "mapping", "same_id", "collision", "symlink"])
def test_preflight_rejects_without_writes(source, tmp_path, bad):
    dest = tmp_path / "Dataset004_Binary"
    if bad == "label":
        write_image(source / "labelsTr/case1.nii.gz", np.full((4, 5, 6), 3, dtype=np.uint8))
    elif bad == "geometry":
        write_image(source / "imagesTr/case0_0000.nii.gz", np.zeros((4, 5, 6)), origin=(0, 0, 0))
    elif bad == "channels":
        (source / "imagesTr/case0_0000.nii.gz").unlink()
    elif bad == "mapping":
        (source / "renaming_map.csv").write_text("nnUNet_case_id,patient_id\ncase0,p0\n")
    elif bad == "same_id":
        dest = tmp_path / "Dataset003_Binary"
    elif bad == "symlink":
        dest.symlink_to(tmp_path / "Dataset005_Missing", target_is_directory=True)
    else:
        (tmp_path / "Dataset004_AlreadyUsed").mkdir()
    with pytest.raises((ValueError, FileExistsError)):
        convert(source, dest)
    assert not dest.exists()


def test_provenance_union_audit(source, tmp_path):
    audit = tmp_path / "originals"
    audit.mkdir()
    raw = np.zeros((4, 5, 6), dtype=np.uint8)
    raw[1, 1, 1], raw[1, 1, 2], raw[2, 1, 1:3] = 1, 3, 2
    write_image(audit / "case0.nii.gz", raw)
    dest = tmp_path / "Dataset004_Binary"
    report = convert(source, dest, audit_dir=audit, dry_run=True)
    assert report["audited_cases"] == 1 and report["unaudited_cases"] == 1
    raw[1, 1, 2] = 0
    write_image(audit / "case0.nii.gz", raw)
    with pytest.raises(ValueError, match="annotation mismatch"):
        convert(source, dest, audit_dir=audit)
    assert not dest.exists()


def test_split_identity_patient_leakage_and_coverage():
    folds = [{"train": ["a"], "val": ["b"]}, {"train": ["b"], "val": ["a"]}]
    validate_splits(folds, {"a": "p1", "b": "p2"})
    with pytest.raises(ValueError, match="Patient leakage"):
        validate_splits(folds, {"a": "p1", "b": "p1"})
    with pytest.raises(ValueError, match="identities"):
        validate_splits(folds, {"a": "p1", "b": "p2", "c": "p3"})
    with pytest.raises(ValueError, match="exactly once"):
        validate_splits([folds[0], folds[0]], {"a": "p1", "b": "p2"})


def test_original_ucsf_layout_and_audit_geometry(source, tmp_path):
    audit = tmp_path / "originals"
    subject = audit / "patient0A"
    subject.mkdir(parents=True)
    raw = np.zeros((4, 5, 6), dtype=np.uint8)
    raw[1, 1, 1], raw[1, 1, 2] = 1, 4
    path = subject / "patient0A_BraTS-seg.nii.gz"
    write_image(path, raw)
    dest = tmp_path / "Dataset004_Binary"
    report = convert(source, dest, audit_dir=audit, dry_run=True)
    assert report["audited_cases"] == 1
    assert report["cases"][0]["audit_path"] == str(path)
    write_image(path, raw, origin=(0, 0, 0))
    with pytest.raises(ValueError, match="same image space"):
        convert(source, dest, audit_dir=audit)
    assert not dest.exists()
