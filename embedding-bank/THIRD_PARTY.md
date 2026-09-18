# Third-party provenance and licenses

This release does not distribute pretrained models or third-party source checkouts.
Registry files specify the source URLs, immutable revisions, downloads, checkpoint variants and known caveats.
The adapter code bridges these upstream APIs; citation and license obligations remain with each original project.

Primary reference implementations:

- [WIBE](https://github.com/ispras/wibe) — legacy wrapper configurations and some weight mirrors.
- [VideoSeal](https://github.com/facebookresearch/videoseal), [TrustMark](https://github.com/adobe/trustmark),
  [VINE](https://github.com/Shilin-LU/VINE), [invisible-watermark](https://github.com/ShieldMnt/invisible-watermark).
- [COCO](https://cocodataset.org/) — source images/captions; image copyright remains with original owners.
- [Diffusers](https://github.com/huggingface/diffusers) — fixed neural channel pipelines.
- Every additional model's official repository and exact commit appear in `registry/*.json`.

Before publishing or reusing model assets, inspect upstream LICENSE/model cards and cite the associated papers.
Some weights/backbones are noncommercial, gated or mirrored. A downloadable checkpoint is not a universal grant.
Do not redistribute author archives simply because a benchmark mirror could fetch them.

The orchestration code is published under the repository's MIT license.
This does not relicense upstream dependencies, model source code or weights.
