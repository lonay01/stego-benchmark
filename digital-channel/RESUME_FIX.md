# Missing-cache resume fix

Notebook: `dist/COCO_Digital_Channel_RESUME_FIX.ipynb`.

The supplied run stopped on the already completed `trustmark_q` method:
`FileNotFoundError: .../attack-cache/<run>/trustmark_q/owner.json`.
It did not reach StegaStamp. The cleanup routine required ownership metadata
even when the disposable archive had already disappeared with the Colab runtime.

This repair moves only the ownership check inside the existing-ZIP deletion
branch. Completed result checks, expected path/filename checks, and ownership
and SHA256 checks before deleting any existing ZIP remain enabled. It neither
creates a replacement owner file nor catches unrelated missing-file errors.

The original source ZIP and previous StegaStamp repair remain unchanged.
The repair is installed separately and applied in memory in the controller.
Only the run entry point and an additional repair cell differ from the previous
notebook. Settings, models, attacks, metrics and durable records are unchanged.

Run all notebook cells in order with the same bank/output paths and
`N_IMAGES = 5`. Stop any previous runner first. The original ZIP is reused
from Drive. Do not remove old results, manifests or `owner.json` manually.

Validation: 54 tests passed (29 original regressions plus 25 repair tests).
The lost-cache condition was reproduced, and a completed method resumed with
zero attack/decode stage calls and unchanged saved-result checksums. All eight
notebook code cells parsed and the notebook schema validated. Present archives
with missing/wrong ownership, changed hashes, foreign paths or incomplete
results remain protected. The new notebook has not been run end-to-end on A100.
See `dist/resume-fix-validation.json`.

Build: `python build_resume_fix.py`.
Validate: `python validate_resume_fix.py` (existing diagnostic dependencies).
