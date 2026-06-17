import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold


def robinson_plm(Y, A, U, n_folds=2, n_estimators=200, seed=42):
    """Robinson (1988) PLM via cross-fitted random forests. Sandwich SE.

    Parameters
    ----------
    Y : ndarray (n_spots, n_genes)
        Outcome matrix in log space.
    A : ndarray (n_spots,)
        Binary treatment indicator (0/1).
    U : ndarray (n_spots, n_features)
        Confounder matrix (spatial PCs + log-library-size).
    n_folds : int
        Number of cross-fitting folds.
    n_estimators : int
        Number of trees in each random forest.
    seed : int
        Random state for reproducibility.

    Returns
    -------
    tau : ndarray (n_genes,)
        Treatment effect estimates.
    se : ndarray (n_genes,)
        Sandwich standard errors.
    """
    n = len(A)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    A_res = np.zeros(n)
    Y_res = np.zeros_like(Y, dtype=np.float64)

    for fold, (train_idx, val_idx) in enumerate(kf.split(U)):
        print(f'  Fold {fold + 1}/{n_folds}...', flush=True)

        clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=seed)
        clf.fit(U[train_idx], A[train_idx])
        A_res[val_idx] = A[val_idx] - clf.predict_proba(U[val_idx])[:, 1]

        reg = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=seed)
        reg.fit(U[train_idx], Y[train_idx])
        Y_res[val_idx] = Y[val_idx] - reg.predict(U[val_idx])

    denom = float(A_res @ A_res)
    tau = (A_res @ Y_res) / denom
    influence = A_res[:, None] * (Y_res - A_res[:, None] * tau)
    se = np.sqrt((influence ** 2).sum(axis=0)) / denom

    return tau, se
