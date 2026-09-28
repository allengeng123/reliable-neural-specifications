# Correct NAP extraction from pretrained FCNs

## Algorithm 1

The paper's Definition 3.1 calls a neuron active exactly when its preactivation
is **strictly positive**. Zero is inactive. Given a class-specific set `S_l`,
Algorithm 1 counts active occurrences `c_j` and chooses:

```text
active neurons:   c_j / |S_l| >= delta
inactive neurons: c_j / |S_l| <= 1 - delta
unconstrained:   everything else
```

Require `0.5 < delta <= 1` to avoid overlapping selected states. This is the
statistical mining algorithm; its per-neuron thresholds do not guarantee
joint pattern recall of `delta`.

Thresholds are interpreted as exact decimals and converted to integer count
cutoffs. For example, at `delta=0.9`, a neuron active in exactly one of ten
samples belongs to the inactive set. Integer cutoffs preserve inclusive
boundaries without floating-point subtraction errors, regardless of how the
dataset is split into batches.

Empirical matching uses `> 0` / `<= 0` consistently with extraction. The
verifier explicitly uses a conservative closed region (`>= 0` / `<= 0`);
that relaxation is separate from empirical NAP membership.

The count accumulator stores integer support and active counts for each class.
Memory for these statistics depends on classes × hidden neurons, not the
dataset size. A single-example class keeps its neuron axis; absent classes are
reported instead of being assigned an unsupported universal pattern.

## Supported pretrained models

| Input | Behavior |
| --- | --- |
| `.nnet` | Any depth/width of affine hidden layers followed by ReLU; float64 inference |
| `.onnx`, `.onnx.gz` | Dense ReLU graphs evaluated by ONNX Runtime at the model's float32/float64 dtype |
| Custom Python adapter | Supply `sizes`, `hidden_size`, and `forward(inputs) -> (scores, preactivations)` |

The ONNX adapter exposes the input and output of every live `Relu` operator
contributing to the score output. It does not rebuild the network from guessed
layer widths, assume module names, or depend on whether the original training
framework used modules or functional ReLU calls. `trace_layout` records tensor
names, widths, and flattened offsets. Terminal linear scores are not counted
as ReLU neurons. The source ONNX file is never modified.

Standard dense exports using `Gemm` or `MatMul`/`Add`, common shape operations,
affine preprocessing, batch normalization, and optional output softmax are
supported. Inputs must have one data tensor, a leading batch dimension of 1 or
dynamic size, static feature dimensions, embedded weights, and one score
output. Evaluation is sample-by-sample so fixed-batch exports work and extraction
batch size does not change the ONNX execution shape.

This means **arbitrary depths and widths of supported pretrained ReLU FCNs**,
not every possible neural architecture or file format. Sigmoid/tanh-only
networks, convolutional networks, control-flow/custom operators, external ONNX
weight files, multi-input models, and fixed batch sizes greater than one are
rejected explicitly. Export a pretrained PyTorch/TensorFlow model in evaluation
mode to ONNX with embedded weights and explicit ReLU operators, or provide a
trace adapter. This package does not load arbitrary pickle checkpoints.

Preactivations are floating-point values from the chosen runtime. A value
extremely close to zero may differ in sign across runtimes or dtypes; no
tolerance is silently applied to change Definition 3.1. Validation records
runtime versions and checks sign agreement on the tested models and inputs.

## Label source and preprocessing

`--label-source provided` groups inputs by dataset labels, corresponding to
`S_l` in the paper. It does not silently remove misclassified inputs.
`--label-source predicted` groups by model predictions. These are different
experiments and are recorded as such.

`--decision argmin` is the ACAS Xu convention. `--decision argmax` is the MNIST
convention. Always state the policy explicitly, even when using dataset labels;
it remains part of the experiment metadata. Do not call prediction agreement
ground-truth accuracy.

`inputs` should already match the model's input space. No dataset-specific
normalization is guessed. For raw physical `.nnet` inputs only, pass
`--input-space physical` to apply header clipping and normalization.
For the official MNIST FC models, use **float32 pixels divided by 255**, with
no extra MNIST mean/std normalization. The model expects `(1, 784, 1)`; the
adapter accepts samples with 784 features and reshapes to that exact layout.
This follows the original benchmark's
[property-generation source](https://github.com/pat676/mnist_fc_vnncomp2022/blob/26878486507c8c5d44bb15e50f5550a9a2b157be/generate_properties.py).

## Runnable official MNIST FC example

Fetch the pinned models/data and prepare small NPZ samples:

```bash
python -m pip install -e '.[onnx,test]'
python scripts/validate_extraction.py fetch
```

Extract a NAP using the training labels:

```bash
python -m neural_specs extract \
  --model benchmarks/cache/mnist-net_256x4.onnx.gz \
  --data benchmarks/cache/mnist-train-1024.npz \
  --label-source provided --decision argmax \
  --delta 0.95 --batch-size 256 --output results/mnist-256x4
```

Substitute `256x2` or `256x6` for the other official models. Results include
`patterns.json` with model/data digests, preprocessing, label policy, class
support and neuron layout, and `counts.npz` with integer sufficient statistics.
Full training datasets can be supplied through the same API; these prepared
1,024 training and 256 test examples provide a small reproducible example.

## Streaming Python API

```python
from neural_specs.extraction import array_batches, extract, load_model

model = load_model("pretrained.onnx")
result = extract(
    model,
    array_batches(inputs, labels, batch_size=256),
    label_source="provided",
    decision="argmax",
    delta="0.95",
)
patterns = result.patterns
```

For a dataset larger than RAM, replace `array_batches(...)` with any iterator
yielding `(input_batch, label_batch)`. Use `(input_batch, None)` for predictions.
`result.counts.patterns("0.9")` can change the threshold without another model
pass. The CLI loads an NPZ array into memory; use the iterator API for genuinely
streaming input storage.

## Validation scope

The reference checks cover all **45 VNN-COMP 2023 ACAS Xu networks** and all
**three VNN-COMP 2022 MNIST FC networks**. Source commits and downloaded asset
SHA-256 values are recorded for reproducibility.

For every model, it checks instrumentation against original graph outputs,
every ReLU's pre/post sign relationship, an independent ONNX reference
interpreter on 16 inputs, Algorithm 1 against an independent scalar oracle at
five thresholds, and identical extraction counts for batch sizes 1, 31, and
256. MNIST checks both dataset-label and model-prediction grouping. It also
compares the bundled `.nnet` ACAS Xu model with its ONNX counterpart and reruns
the original small Marabou demonstration. See the recorded validation results
in [validation notes](validation.md).

The suite tests **extraction correctness and portability**. It does not assert
that every extracted NAP has high precision, high recall, or a successful
formal robustness proof; those are separate empirical/verification questions.
