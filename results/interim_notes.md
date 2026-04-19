# Interim Experimental Notes

This file tracks raw observations, failed attempts, and intermediate
metric readings during the course of the project.  It is not polished
writing — see README.md Part 3 for the formatted interim report.

---

## Baseline CNN (ReLU) — initial run

| Epochs | LR     | Val Acc |
|--------|--------|---------|
| 15     | 1e-3   | TBD     |

Notes:
- Training was stable throughout.
- Loss converged by epoch 10.

---

## Square Activation CNN — initial run

| Epochs | LR     | Clip Norm | Val Acc |
|--------|--------|-----------|---------|
| 20     | 1e-4   | 1.0       | TBD     |
| 20     | 5e-5   | 0.5       | TBD     |

Notes:
- Gradient explosion observed in first attempt (LR=1e-3, no clipping).
- BatchNorm stabilised training after adding weight decay.
- Symmetry issue: model initially predicts same class for many inputs
  because x^2 cannot distinguish sign; BatchNorm partially mitigates this.

---

## Simulated FHE accuracy (quantisation proxy)

| Bit-width | Approx Acc |
|-----------|------------|
| 16-bit    | TBD        |
| 8-bit     | TBD        |
| 6-bit     | TBD        |
| 4-bit     | TBD        |

---

## Open items

- [ ] Decide between TenSEAL and Concrete-ML
- [ ] Resolve Docker requirement for Concrete-ML on Windows
- [ ] Measure actual multiplicative depth of compiled circuit
- [ ] Profile inference latency per image
