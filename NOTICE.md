# Attribution and licensing

This repository accompanies **Towards Reliable Neural Specifications**,
ICML 2023 Oral, by Chuqin Geng, Nham Le, Xiaojie Xu, Zhaoyue Wang,
Arie Gurfinkel, and Xujie Si.

[Paper](https://icml.cc/virtual/2023/oral/25466)

## Research code

Related research implementations include
[VerifyNNE](https://github.com/allengeng123/VerifyNNE/tree/7f71d13ab945f8b6ba4c90b640b42d64437d9dbe)
(a fork of `nhamlv-55/VerifyNNE`) and
[Verify-Network](https://github.com/allengeng123/Verify-Network/tree/55d850c994f9f6539f4ed9d2731f4eab380b438d).
No repository-wide software license is currently specified.

## Models and dependencies

The bundled ACAS Xu `.nnet` model carries a **CC BY 4.0** notice and credits
the format to **Kyle Julian, Stanford, 2016**. The notice is preserved in the
model file. Its source, license link, and checksum are in
[the model documentation](data/acasxu/README.md).

VNN-COMP reference models and MNIST data are downloaded separately by the
validation script. Their source URLs and checksums are recorded with the
results; they are not bundled in this repository.

Marabou, NumPy, ONNX, ONNX Runtime, pytest, and build tooling are external
dependencies distributed under their own licenses.
