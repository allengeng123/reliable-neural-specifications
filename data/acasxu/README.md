# Bundled ACAS Xu model

`ACASXU_run2a_1_1_batch_2000.nnet` is copied without modification from:

[VerifyNNE, commit 7f71d13ab945f8b6ba4c90b640b42d64437d9dbe](https://github.com/allengeng123/VerifyNNE/blob/7f71d13ab945f8b6ba4c90b640b42d64437d9dbe/datasets/ACAS/acas_nets/ACASXU_run2a_1_1_batch_2000.nnet).

SHA-256: `b79d4bc4056ab44a8985ad2e3eed646a20610422c913367f63060f7680573ce3`

The header attributes the Neural Network File Format to **Kyle Julian,
Stanford, 2016**, and licenses the file contents under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
That notice is preserved verbatim. No model weights were retrained or modified.

Architecture: five inputs, six hidden layers of 50 ReLUs, five outputs. The
output advisory with the lowest score wins: clear of conflict, weak left,
weak right, strong left, strong right (indices 0 through 4).

No serialized Python/PyTorch training datasets are needed. The demo creates
small deterministic samples at runtime from the model's declared domain.
