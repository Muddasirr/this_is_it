# Vendored: PARSeq (inference only)

- Upstream: https://github.com/baudm/parseq
- Commit: `1902db043c029a7e03a3818c616c06600af574be` (2024-04-24)
- Licence: Apache-2.0 (`LICENSE`, unchanged). Copyright 2022 Darwin Bautista.
- Weights: release `v1.0.0/parseq-bb5792a6.pt`, SHA-256
  `bb5792a68e367476abca029cbf8699abc805f3d3dc7e57aae45c8ec4f7b7cd00` (checked in `load()`),
  stored in our own storage (`backend/models/`, Modal volume), never fetched at runtime.

| File here | Upstream | Change |
|---|---|---|
| `model.py` | `strhub/models/parseq/model.py` | imports point at this package |
| `modules.py` | `strhub/models/parseq/modules.py` | none |
| `tokenizer.py` | `strhub/data/utils.py` | none |
| `weights.py` | `strhub/models/utils.py` | `init_weights` only |
| `__init__.py` | — | ours: pinned config, checksum, loader, preprocessing |

Not vendored (training only): Lightning `system.py`, data modules, augmentation, `nltk`.
To update: diff the upstream files at a new commit, review, re-run `tests/test_parseq_vendor.py`.
