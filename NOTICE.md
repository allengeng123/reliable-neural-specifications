# Attribution and licensing status

This repository accompanies **Towards Reliable Neural Specifications**,
ICML 2023 Oral, by Chuqin Geng, Nham Le, Xiaojie Xu, Zhaoyue Wang,
Arie Gurfinkel, and Xujie Si.

Paper: https://icml.cc/virtual/2023/oral/25466

Historical implementations reviewed for this focused reorganization:

- `allengeng123/VerifyNNE` (fork of `nhamlv-55/VerifyNNE`), commit
  `7f71d13ab945f8b6ba4c90b640b42d64437d9dbe`.
- `allengeng123/Verify-Network`, commit
  `55d850c994f9f6539f4ed9d2731f4eab380b438d`.

Neither historical snapshot contains a repository-level license file. This
cleanup does not assign a new license to the historical code or claim to
relicense it. No repository-wide software license has been selected for this
edition; authors may add one separately.

The bundled `.nnet` model carries an explicit **CC BY 4.0** notice and credits
the format to **Kyle Julian, Stanford, 2016**. It is copied byte-for-byte from
the pinned VerifyNNE source; its provenance, license link, and checksum appear
in `data/acasxu/README.md`.

Marabou, NumPy, pytest, and build tooling are external dependencies distributed
under their own licenses. Marabou is not vendored here.
