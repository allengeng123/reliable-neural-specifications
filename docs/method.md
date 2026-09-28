# Method and conventions

## Mining

For each predicted class, count the fraction `p` of samples for which a hidden
neuron's preactivation is strictly positive. Select an active state if
`p >= delta` and an inactive state if `p <= 1 - delta`, with
`0.5 < delta <= 1`. Other neurons remain unconstrained. This implements the
frequency rule in Algorithm 1 of the paper. The actual joint pattern coverage
is measured separately; per-neuron frequencies do not imply joint recall of
at least `delta`.

Neurons are indexed in hidden-layer order, then within each layer. Pattern files
store binary states, positive class support, model SHA-256, and architecture.
A class absent from the mining samples is omitted and reported. A class with
support but no stable neurons legitimately has an unconstrained pattern.

The ACAS demonstration uses synthetic inputs drawn uniformly from the `.nnet`
domain and model-predicted labels. It measures agreement with the network,
not real-world classification accuracy or accuracy against pilot advisories.
Training and held-out samples are disjoint draws from one seeded generator.

## Coordinates and activation boundaries

Raw physical coordinates are normalized with `(input - mean) / range` using the
model header. The CLI samples and verifies directly in normalized coordinates.
Output scores are also normalized. Shared positive output scaling preserves
the argmin. The declared physical bounds are never replaced by a generic `[-1,1]`.

For verification, active means `preactivation >= 0` and inactive means
`preactivation <= 0`. These closed half-spaces deliberately overlap at zero,
giving a conservative superset of a strictly positive active pattern. This
does not silently remove boundary inputs to make a proof easier.
`--activation-margin m` explicitly restricts active/inactive neurons to
`>= m` / `<= -m`; a positive margin changes the verified region and is recorded.

## Verification

For a target advisory `k`, the property is:

```text
for every x in the stated input box satisfying the NAP,
    score[k](x) + output_margin < score[j](x), for every j != k.
```

Marabou first checks feasibility of the input box intersected with the pattern.
Only a feasible region can be called verified. Each competing class then gets
a fresh network with the negated property:

```text
score[k] - score[j] >= -output_margin
```

All these queries must return UNSAT to certify strict dominance. A SAT tie
violates this property even when NumPy's argmin tie-breaking returns `k`.
SAT inputs are independently evaluated in NumPy and checked against the input
bounds, NAP, and output inequality with a reported numerical tolerance of 1e-6.
An invalid witness, unknown, or timeout remains inconclusive. A counterexample
to one competitor is sufficient to disprove the property even if another
comparison times out.

Marabou is a numerical solver; an UNSAT result here is solver-backed
verification under its numerical semantics, not an independently checked
exact-arithmetic proof certificate. Sample-based coverage is never used as a
substitute for verification.

## Small demo versus full-domain query

The default demo uses the first held-out input matching its predicted-class
NAP and intersects that NAP with a radius-0.001 box. It verifies all four
competing advisories in that small region. The `verify` command instead uses
the full domain. The demo does not establish that the NAP improves the
verified radius over an unconstrained baseline, nor reproduce the paper's
MNIST/CIFAR-10 results.
