"""
Smoke test for the TIDEST package.

Runs the full public pipeline on a small synthetic dataset (no R, no downloads)
and asserts that TIDEST recovers the DE genes with controlled false positives.

Run from the repository root:

    pytest tests/
"""

import os
import sys

import numpy as np

# Make examples/_synthetic.py importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

from tidest import tidest          # noqa: E402
from _synthetic import make_dataset  # noqa: E402


def test_import():
    """The package imports and exposes the estimator class."""
    assert callable(tidest)


def test_end_to_end_recovers_de_genes():
    """TIDEST recovers most DE genes while keeping the FPR low."""
    data = make_dataset(N=300, G_DE=30, G_null=70, M=10, n_pcs=20, seed=0)

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

    res = model.results_.set_index("gene")
    assert len(res) == len(data["genes"])
    assert res["qval"].notna().all()

    is_de = dict(zip(data["genes"], data["is_de"]))
    res = res.assign(true_de=[is_de[g] for g in res.index])
    sig = res["qval"] < 0.05

    n_de = int(res["true_de"].sum())
    n_null = int((~res["true_de"]).sum())
    tpr = (sig & res["true_de"]).sum() / n_de
    fpr = (sig & ~res["true_de"]).sum() / n_null

    # Power should be high and false positives controlled on this easy setting.
    assert tpr >= 0.7, f"power too low: {tpr:.2f}"
    assert fpr <= 0.15, f"FPR too high: {fpr:.2f}"

    # Detected DE genes should have the correct sign.
    tau_true = dict(zip(data["genes"], data["tau_true"]))
    det = res[sig & res["true_de"]]
    correct = np.sign(det["tau"]) == [np.sign(tau_true[g]) for g in det.index]
    assert correct.mean() >= 0.9


if __name__ == "__main__":
    # Allow running without pytest installed: invoke the test functions directly.
    try:
        import pytest
        raise SystemExit(pytest.main([__file__, "-v"]))
    except ImportError:
        test_import()
        test_end_to_end_recovers_de_genes()
        print("All smoke tests passed.")
