# What moved, and why

The source repositories remain unchanged. This is a focused, newly structured
implementation of their ACAS Xu/NAP workflow rather than a Git history merge.

| Historical source | Role here |
| --- | --- |
| `VerifyNNE/AcasXuNet.py` | `.nnet` inference and normalization are now `src/neural_specs/nnet.py`, using float64 NumPy |
| `VerifyNNE/BaseNet.py`, `ReluPatterns.py`, `playground.ipynb` | Explicit activation traces and frequency-based mining in `patterns.py` |
| `VerifyNNE/marabou_acas.py` | Bounded, import-safe Marabou queries in `verification.py` |
| `VerifyNNE/run_acas.py` | Portable CLI with model paths, seeds, and JSON reports |
| `Verify-Network/marabou_mnist*.py` | Historical related NAP verification scripts; MNIST execution is intentionally out of scope |
| `datasets/ACAS/acas_nets` | One original `.nnet` model included byte-for-byte with checksum |

Correctness and usability changes:

- Removed absolute author-machine paths and script execution during import.
- Removed the PyTorch/ONNX/pandas dependency chain from the small ACAS workflow.
- Replaced hard-coded width-50 variable offsets with Marabou's layer/node mapping.
- Store binary active/inactive states rather than ambiguous raw counts.
- Use each model's actual normalized domain and ACAS argmin convention.
- Check feasibility separately to avoid vacuous verification of empty regions.
- Keep SAT, UNSAT, unknown, and timeout distinct; do not index missing assignments
  after UNSAT or mask solver errors with a broad exception handler.
- Bind patterns to the exact model digest and reject malformed patterns.
- Keep the sample axis for single-example classes and report missing classes.
- Use Windows-safe paths and regenerate small seeded samples rather than copying
  the legacy `AcasNetID<1,1>-*.pt` files.

Excluded from this edition: vendored image datasets and large experiments, old generated
results, notebooks, TinyAbsInt research scaffolding, all 45-network sweeps,
training, and data-dependent absolute filesystem paths. The original links
preserve access to historical material.

The subsequent extraction audit adds an optional downloader for the three
official MNIST FC models and small MNIST samples, plus all 45 ACAS Xu ONNX
models. These assets remain outside Git in `benchmarks/cache/`. Extraction is
now model-independent; formal verification remains ACAS Xu focused. See
[the extraction guide](extraction.md) for the corrected Algorithm 1 semantics.

## Pinned historical sources

- [VerifyNNE at 7f71d13](https://github.com/allengeng123/VerifyNNE/tree/7f71d13ab945f8b6ba4c90b640b42d64437d9dbe)
- [Verify-Network at 55d850c](https://github.com/allengeng123/Verify-Network/tree/55d850c994f9f6539f4ed9d2731f4eab380b438d)

`VerifyNNE` is a fork of `nhamlv-55/VerifyNNE`; its `.nnet` reader credits
[XuankangLin/ART](https://github.com/XuankangLin/ART). The new reader is implemented
directly against the `.nnet` format. See the model notice and repository NOTICE.
