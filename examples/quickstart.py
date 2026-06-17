"""
TIDEST quick-start example (synthetic, self-contained).

Runs the full TIDEST pipeline on a small synthetic spatial-transcriptomics
dataset in a few seconds. No data download and no R / SpatialPCA install are
needed: a precomputed spatial-PC matrix ``U`` is supplied to ``fit`` so the
R-based confounder step is bypassed.

Run from the repository root:

    python examples/quickstart.py
"""

import numpy as np

from tidest import tidest
from _synthetic import make_dataset


def main():
    # 1. Build a synthetic dataset: 300 spots, 30 DE genes + 70 null genes.
    data = make_dataset(N=400, G_DE=30, G_null=70, M=10, n_pcs=20, seed=0)

    # 2. Fit TIDEST via the public API.
    #    - sc_adata  -> gene-gene Pearson matrix for the augmented outcome
    #    - st_adata  -> observed counts + spatial coordinates
    #    - pred_adata-> imputed expression (here pred_is_log=True, like CellPLM)
    #    - U         -> precomputed spatial confounders (skips SpatialPCA / R)
    #    - treatment -> passed as an array, so no obs column / treatment_val needed
    model = tidest(n_pcs=20, n_folds=2, corr_threshold=0.5, seed=0)
    model.fit(
        sc_adata=data["sc_adata"],
        st_adata=data["st_adata"],
        pred_adata=data["pred_adata"],
        U=data["U"],
        treatment=data["A"],
        genes=data["genes"],
        pred_is_log=True,
    )

    res = model.results_
    print("\nTop 10 genes by p-value:")
    print(res.head(10).to_string(index=False))

    # 3. Evaluate against the known ground truth.
    is_de = dict(zip(data["genes"], data["is_de"]))
    res = res.assign(true_de=res["gene"].map(is_de))
    sig = res["qval"] < 0.05

    n_de = int(res["true_de"].sum())
    n_null = int((~res["true_de"]).sum())
    tpr = (sig & res["true_de"]).sum() / max(n_de, 1)
    fpr = (sig & ~res["true_de"]).sum() / max(n_null, 1)

    print(f"\nDetected {int(sig.sum())} / {len(res)} genes at q < 0.05")
    print(f"True-positive rate (power):  {tpr:.2f}  ({n_de} DE genes)")
    print(f"False-positive rate:         {fpr:.2f}  ({n_null} null genes)")

    # Sign accuracy on detected DE genes
    detected_de = res[sig & res["true_de"]]
    tau_true = dict(zip(data["genes"], data["tau_true"]))
    correct_sign = np.sign(detected_de["tau"]) == detected_de["gene"].map(
        lambda g: np.sign(tau_true[g]))
    if len(detected_de):
        print(f"Sign accuracy on detected DE genes: "
              f"{correct_sign.mean():.2f} ({len(detected_de)} genes)")

    return res


if __name__ == "__main__":
    main()
