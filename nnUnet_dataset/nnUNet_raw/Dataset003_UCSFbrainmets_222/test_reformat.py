"""Standard-library integration checks; temporary byte fixtures, no clinical data."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
COHORT = HERE / 'ucsf_scan_A_preoperative_brats_training_222.csv'


class ReformatTest(unittest.TestCase):
    def test_cohort_and_failure_modes(self):
        with tempfile.TemporaryDirectory(prefix='ucsf222-test-') as temp:
            root = Path(temp)
            classes = root / 'classes'
            classes.mkdir()
            subprocess.run(['javac', '-d', str(classes), str(HERE / 'reformat.java')], check=True)
            source = root / 'source with spaces,comma'
            source.mkdir()
            target = root / 'output'
            with COHORT.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(222, len(rows))
            self.assertEqual(222, len({r['patient_id'] for r in rows}))
            self.assertEqual(72, sum(int(r['total_scans_in_source']) > 1 for r in rows))
            for r in rows:
                for suffix in ['FLAIR_BC', 'T1ce_BC', 'combined.seg']:
                    folder = source / r['subject_id']
                    folder.mkdir(exist_ok=True)
                    (folder / f'{r["subject_id"]}_{suffix}.nii.gz').write_bytes(f'{r["subject_id"]}:{suffix}'.encode())
            for sid in ['100108B', '100108C', '100414A', '100414B', '999999A']:
                for suffix in ['FLAIR', 'T1post', 'combined.seg']:
                    (source / f'{sid}_{suffix}.nii.gz').write_bytes(b'EXCLUDED')

            def run(ok=True, cohort=COHORT, dry=False, dest=target):
                cmd = ['java', '-cp', str(classes), 'reformat', str(source), str(dest), str(cohort)]
                result = subprocess.run(cmd + (['--dry-run'] if dry else []), capture_output=True, text=True)
                self.assertEqual(ok, result.returncode == 0, result.stdout + result.stderr)
                return result.stdout + result.stderr

            def snapshot():
                return {str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in target.rglob('*') if p.is_file()} if target.exists() else {}

            run(dry=True)
            self.assertFalse(target.exists(), 'dry-run created output')
            # All source preflight failures must happen before any target is created.
            first = source / rows[0]['subject_id']
            label = first / f'{rows[0]["subject_id"]}_combined.seg.nii.gz'
            original = label.read_bytes()
            label.unlink()
            self.assertIn('expected one file', run(ok=False))
            self.assertFalse(target.exists())
            label.write_bytes(b'')
            self.assertIn('empty', run(ok=False))
            label.write_bytes(original)
            duplicate = source / label.name
            duplicate.write_bytes(original)
            self.assertIn('expected one file', run(ok=False))
            duplicate.unlink()
            duplicate = first / f'{rows[0]["subject_id"]}_FLAIR.nii.gz'
            duplicate.write_bytes(b'ambiguous channel')
            run(ok=False)
            duplicate.unlink()
            self.assertFalse(target.exists())

            bad = root / 'bad.csv'
            def write_rows(data, fields=None):
                with bad.open('w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
                    w.writeheader()
                    w.writerows(data)
            for field, value in [('patient_id', '100414'), ('subject_id', '100101B'),
                                 ('scan_letter', 'B'), ('prior_craniotomy_biopsy_resection', 'Yes'),
                                 ('brats_id', 'BraTS-MET-00553-001')]:
                altered = [dict(r) for r in rows]
                altered[0][field] = value
                write_rows(altered)
                run(ok=False, cohort=bad)
            write_rows(rows[:-1]); run(ok=False, cohort=bad)
            write_rows(rows[:-1] + [rows[0]]); run(ok=False, cohort=bad)
            bad.write_text('patient_id,subject_id\n"unterminated')
            run(ok=False, cohort=bad)
            bad.write_text('patient_id,patient_id\na,b\n')
            run(ok=False, cohort=bad)
            bad.write_text('patient_id,subject_id\n100101,100101A\n')
            run(ok=False, cohort=bad)
            # Reordered headers/rows, BOM, CRLF, quoted commas and escaped quotes.
            altered = [dict(r, note='comma, quote " and\nnewline') for r in reversed(rows)]
            write_rows(altered, ['note'] + list(reversed(list(rows[0]))))
            bad.write_bytes(b'\xef\xbb\xbf' + bad.read_bytes())
            run(cohort=bad, dry=True)
            self.assertFalse(target.exists())

            run()
            self.assertEqual(444, len(list((target / 'imagesTr').iterdir())))
            self.assertEqual(222, len(list((target / 'labelsTr').iterdir())))
            metadata = json.loads((target / 'dataset.json').read_text())
            self.assertEqual(222, metadata['numTraining'])
            self.assertEqual({'0': 'FLAIR', '1': 'T1post'}, metadata['channel_names'])
            self.assertEqual({'background': 0, 'tumor_core': 1, 'edema': 2}, metadata['labels'])
            with (target / 'renaming_map.csv').open() as stream:
                mapping = list(csv.DictReader(stream))
            self.assertEqual(sorted(r['subject_id'] for r in rows), [r['SubjectID'] for r in mapping])
            for i, r in enumerate(mapping):
                self.assertEqual(f'UCSFbrainmets_{i:03d}', r['nnUNet_case_id'])
                self.assertEqual(r['SubjectID'][:-1], r['patient_id'])
                for col, folder, suffix in [('FLAIR', 'imagesTr', '_0000'), ('T1post', 'imagesTr', '_0001'), ('Label', 'labelsTr', '')]:
                    self.assertEqual(Path(r[col]).read_bytes(), (target / folder / (r['nnUNet_case_id'] + suffix + '.nii.gz')).read_bytes())
            before = snapshot()
            run(dry=True); self.assertEqual(before, snapshot())
            run(); self.assertEqual(before, snapshot())
            # Interrupted copy: intact identity map permits replacing missing outputs.
            (target / 'labelsTr/UCSFbrainmets_010.nii.gz').unlink()
            run(); self.assertEqual(before, snapshot())
            extra = target / 'imagesTr/extra.nii.gz'
            extra.write_bytes(b'extra')
            changed = snapshot(); run(ok=False); self.assertEqual(changed, snapshot()); extra.unlink()
            mapfile = target / 'renaming_map.csv'
            maptext = mapfile.read_text()
            mapfile.write_text(maptext.replace('100101A', '100101B'))
            changed = snapshot(); run(ok=False); self.assertEqual(changed, snapshot()); mapfile.write_text(maptext)
            mapfile.unlink(); run(ok=False); mapfile.write_text(maptext)
            original = label.read_bytes(); label.write_bytes(b'changed source')
            changed = snapshot(); run(ok=False); self.assertEqual(changed, snapshot()); label.write_bytes(original)
            output_label = target / 'labelsTr/UCSFbrainmets_000.nii.gz'
            original = output_label.read_bytes(); output_label.write_bytes(b'corrupt output')
            changed = snapshot(); run(ok=False); self.assertEqual(changed, snapshot()); output_label.write_bytes(original)
            run(ok=False, dest=source / 'nested-output')
            self.assertFalse((source / 'nested-output').exists())
            jsonfile = target / 'dataset.json'
            original = jsonfile.read_text(); jsonfile.write_text('{}')
            run(ok=False); jsonfile.write_text(original)
            # Default CSV lookup and Java source-file invocation.
            (target / COHORT.name).write_bytes(COHORT.read_bytes())
            result = subprocess.run(['java', str(HERE / 'reformat.java'), str(source), str(target), '--dry-run'], capture_output=True, text=True)
            self.assertEqual(0, result.returncode, result.stderr)


if __name__ == '__main__':
    unittest.main()
