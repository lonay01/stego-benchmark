# SepMark byte-identical recovery — 16 October 2026

The digital runner stopped while checking the existing bank. Three Native PNGs
had zero-byte contents; their sealed records still contained the original hashes.
This was an artifact-storage failure, not an observed watermark decoding failure.
The underlying cause of the original empty-file writes has not been established.

## Verified repair

| Native file | Restored bytes | Original SHA256 | Clean BER |
|---|---:|---|---:|
| `000000383838_m01.png` | 106264 | `a44eb26509302480b12669fa55f202a952cb5bbf839e2d1df82685c64b75f452` | 0 |
| `000000419882_m04.png` | 111796 | `55b14d801998b3c781f5643b37cb7ae9568d8495ca636143bb95c89429af4e43` | 0 |
| `000000419882_m05.png` | 111722 | `3a48b66dc8868b837e3cbf7ee46cc6b1c2a1a0555bb53cc07b9a882d1008ac04` | 0 |

All three candidates reproduced the old PNG bytes exactly. Blind extraction also
reproduced all 128 saved decoded bits for each case. The frozen metadata, expected
hashes, matched-quality carriers and digital-channel measurements were not changed.
Only the three zero-byte Native files were replaced. Copies of the empty originals
were kept in the recovery directory. Mounted-Drive readback verified the hashes;
the separate Drive connector then confirmed the new nonzero cloud file sizes.

The recovery used the original source revision, checkpoint, dependency lock and
payloads, with Python 3.10.18, PyTorch 2.2.2+cu121, cuDNN 8902, A100-SXM4-80GB and
the recorded per-case seeds. It did not search for improved BER or change the
quantization/matching policy.

## Evidence and tooling

- Cloud audit: <https://drive.google.com/drive/folders/1AGV_oYqKr5Z0rYJkmfjRvE-3tSXFaoKQ>
- `sepmark_byte_recovery.py`: candidates and commit are separate actions. The
  commit refuses a changed candidate, changed sealed record or nonempty changed
  destination. Candidate hashes are checked against the original fixed hashes.
- `test_sepmark_byte_recovery.py`: five local transaction tests passed. These
  tests use synthetic bytes and do not claim model inference; the real inference
  was performed separately in Colab.
- Cloud `candidates_v1/candidate_report.json`, `original_record_hashes.json`,
  `restore_receipt_*.json`, `prepare.log`, `candidates.log` and `restore.log` retain
  the execution evidence.

The one-time Colab setup/generation/commit cells were replaced with no-op notes
after success so future **Run all** does not repeat the repair. Original runner
cells were not edited. The full digital benchmark was **not restarted** during
this recovery. This repair does not certify yet-unprocessed methods or files.
