"""
Full competitor comparison for the GBM (CT vs LE) application — fitted
PER SAMPLE on each of the 26 Visium sections, then pooled across samples
with the SAME DerSimonian-Laird random-effects (DL-RE) meta-analysis used
for tidest (gbm_meta_analysis.py), so that all five methods are compared
on a like-for-like footing:

  1. tidest      — Robinson PLM (per-sample tau/se from gbm_plm_results_all.csv)
  2. t-test      — Welch t-test on log-CPM observed counts (native diff + SE)
  3. SpaGCN      — Wilcoxon rank-sum on log-CPM observed counts (signed z; SE=1)
  4. DESpace     — edgeR NB via svg_test (R subprocess; logFC, SE derived from p)
  5. SpatialGEE  — GEE Poisson via run_gee_gst (R subprocess; coef, SE derived from p)

For methods that natively report only an effect estimate and a p-value
(DESpace, SpatialGEE), the per-sample standard error is back-derived as
se = |effect| / |Phi^-1(1 - p/2)|, the standard approach for meta-analysing
studies that report point estimates and p-values without CIs/SEs. For
SpaGCN/Wilcoxon, the rank-sum z-statistic is itself the standardized effect
(SE = 1), so DL-RE pooling of z-scores is equivalent to a random-effects
Stouffer combination.

Outputs:
  results_raw/gbm_method_comparison_full.csv     (per-method, per-gene pooled results)
  results_raw/gbm_method_comparison_full_summary.csv (per-method aggregate scoring)
"""

import os, sys, warnings, subprocess, tempfile
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

warnings.filterwarnings('ignore')

_ROOT    = os.path.dirname(os.path.abspath(__file__ + '/..'))
_SIM_DIR = os.path.join(_ROOT, 'scripts', 'simulation')

ALPHA        = 0.05
META_PATH    = './Inputs/general/visium_metadata.csv'
PLM_ALL      = './results_raw/gbm_plm_results_all.csv'
OUT_PERGENE  = './results_raw/gbm_method_comparison_full.csv'
OUT_SUMMARY  = './results_raw/gbm_method_comparison_full_summary.csv'

SAMPLES = [
    'MGH258',
    'UKF243', 'UKF248', 'UKF251', 'UKF255', 'UKF259', 'UKF260',
    'UKF266', 'UKF269', 'UKF275', 'UKF296', 'UKF304', 'UKF313', 'UKF334',
    'ZH1007inf', 'ZH1007nec', 'ZH1019inf', 'ZH1019T1',
    'ZH8811Abulk', 'ZH8811Bbulk', 'ZH8812bulk',
    'ZH881inf', 'ZH881T1',
    'ZH916bulk', 'ZH916inf', 'ZH916T1',
]

MARKER_GENES = [
    'top2a','mki67','pcna','mcm2','mcm6','cd44','tnc','chi3l1','fn1','vim',
    'tgfbi','col1a1','serpine1','apoe','sparcl1','aldoc','slc1a2','glul','gja1',
    'olig1','olig2','pdgfra','sox10','sox11','ascl1','dcx','map2','tubb3',
    'mmp9','mmp2','plaur','itgb1','cxcr4','mbp','mog','mag','plp1',
    'syt1','snap25','rbfox3','gfap','aqp4','s100b',
    'aif1','cd68','tmem119','p2ry12','pecam1','vwf','cldn5',
    'hif1a','vegfa','ldha','ca9','egfr','met',
    'hexb','sall1','olfml3','cx3cr1','spp1','lgals1','lgals3','cd163','msr1','mrc1',
    'ptprz1','nrcam','ckb','htra1','anxa2','postn','bcan','vcan','l1cam',
    'sox4','egr1','ccnd2','nes','ldhb','hk2','epas1','stat3','il6',
    'nrxn1','dlg4','ptpn11','idh1','cdkn2a','pten',
]

# Expected direction: +1 = LE-enriched, -1 = CT-enriched, 0 = no directional prior
EXPECTED = {
    **{g: +1 for g in ['syt1','snap25','rbfox3','tubb3','mbp','plp1','mag',
                        'mmp9','mmp2','plaur','itgb1','cxcr4','tmem119','p2ry12',
                        'aif1','cd68','pecam1','vwf','cldn5','ldha',
                        'olig1','olig2','ascl1','dcx','sox10','sox11','pdgfra',
                        'map2','hexb','sall1','olfml3','cx3cr1','ptprz1','nrcam','bcan',
                        'l1cam','htra1','sox4','egr1','ccnd2','ldhb','nrxn1','dlg4','ckb']},
    **{g: -1 for g in ['top2a','mki67','pcna','mcm2','mcm6','egfr','met','mog',
                        'cd44','fn1','vim','tgfbi','col1a1','serpine1','tnc','chi3l1',
                        'apoe','sparcl1','aldoc','glul','gja1','slc1a2',
                        'hif1a','vegfa','ca9','gfap','aqp4','s100b',
                        'spp1','lgals1','lgals3','cd163','msr1','mrc1',
                        'vcan','anxa2','postn','hk2','epas1','il6','cdkn2a']},
    **{g:  0 for g in ['nes','stat3','ptpn11','idh1','pten']},
}


# ── DerSimonian-Laird random-effects meta-analysis (matches gbm_meta_analysis.py) ──
def dersimonian_laird(tau_vec, se_vec):
    tau_vec = np.asarray(tau_vec, dtype=np.float64)
    se_vec  = np.asarray(se_vec,  dtype=np.float64) + 1e-12
    k    = len(tau_vec)
    w_fe = 1.0 / se_vec**2
    tau_fe = (w_fe * tau_vec).sum() / w_fe.sum()
    Q      = (w_fe * (tau_vec - tau_fe)**2).sum()
    c      = w_fe.sum() - (w_fe**2).sum() / w_fe.sum()
    tau2   = max(0.0, (Q - (k - 1)) / c) if c > 1e-300 else 0.0
    w_re   = 1.0 / (se_vec**2 + tau2)
    tau_re = (w_re * tau_vec).sum() / w_re.sum()
    se_re  = 1.0 / np.sqrt(w_re.sum())
    z_re   = tau_re / se_re
    p_re   = 2 * stats.norm.sf(abs(z_re))
    return tau_re, se_re, z_re, p_re


def se_from_effect_and_pvalue(effect, pval):
    """Back-derive an approximate SE from a point estimate and two-sided p-value."""
    pval = np.clip(pval, 1e-300, 1 - 1e-16)
    z = stats.norm.isf(pval / 2.0)
    if z <= 0 or not np.isfinite(z) or effect == 0 or not np.isfinite(effect):
        return np.nan
    return abs(effect) / z


def bh_correct(pvals):
    pvals = np.asarray(pvals, dtype=np.float64)
    mask  = np.isfinite(pvals)
    q = np.full(len(pvals), np.nan)
    if mask.sum():
        _, qv, _, _ = multipletests(pvals[mask], method='fdr_bh')
        q[mask] = qv
    return q


def run_r_method(rscript, C_obs_int, A, coords, gene_names, spot_names, timeout=900):
    """Run a simulation R competitor script on one sample's data; returns per-gene
    (effect, pvalue) where effect is the method's native statistic (logFC / GEE coef)."""
    G = C_obs_int.shape[1]
    with tempfile.TemporaryDirectory() as tmp:
        counts_f = os.path.join(tmp, 'counts.csv')
        meta_f   = os.path.join(tmp, 'meta.csv')
        out_f    = os.path.join(tmp, 'results.csv')
        pd.DataFrame(C_obs_int.T, index=gene_names, columns=spot_names).to_csv(counts_f)
        pd.DataFrame({'spot_id': spot_names, 'A': A,
                      'x': coords[:, 0], 'y': coords[:, 1]}).to_csv(meta_f, index=False)
        res = subprocess.run(['Rscript', '--vanilla', rscript, counts_f, meta_f, out_f],
                             capture_output=True, text=True, timeout=timeout)
        if res.returncode != 0 or not os.path.exists(out_f):
            raise RuntimeError(f'{os.path.basename(rscript)} failed:\n{res.stderr[-1000:]}')
        r_df = pd.read_csv(out_f)

    g2i    = {g: i for i, g in enumerate(gene_names)}
    pval   = np.full(G, np.nan)
    effect = np.full(G, np.nan)
    eff_col = 'logfc' if 'logfc' in r_df.columns else ('gee_coef' if 'gee_coef' in r_df.columns else None)
    for _, row in r_df.iterrows():
        idx = g2i.get(str(row['gene']))
        if idx is not None:
            if not np.isnan(float(row['pvalue'])):
                pval[idx] = float(row['pvalue'])
            if eff_col is not None and not np.isnan(float(row[eff_col])):
                effect[idx] = float(row[eff_col])
    return effect, pval


# ── Load metadata + pre-computed PLM results ───────────────────────────────────
meta_all = pd.read_csv(META_PATH)
plm_all  = pd.read_csv(PLM_ALL)
plm_all['gene'] = plm_all['gene'].str.lower()

rscript_despace    = os.path.join(_SIM_DIR, 'run_despace.R')
rscript_spatialgee = os.path.join(_SIM_DIR, 'run_spatialgee.R')

# Per-sample, per-gene records: method -> list of {sample, gene, tau, se, pval}
records = {m: [] for m in ['t-test', 'SpaGCN', 'DESpace', 'SpatialGEE']}

print(f'Fitting competitors per sample on {len(SAMPLES)} GBM Visium sections...')
for sample in tqdm(SAMPLES, desc='samples'):
    adata = sc.read_visium(f'./Inputs/general/GBM_data/{sample}/outs/')
    adata.var_names_make_unique()
    adata.var_names = pd.Index(adata.var_names.str.lower())
    meta_s = meta_all[meta_all['sample'] == sample].set_index('spot_id')
    ct_le  = meta_s[meta_s['ivygap'].isin(['CT', 'LE'])].index
    adata  = adata[adata.obs_names.intersection(ct_le)].copy()
    ivygap = meta_s.loc[adata.obs_names, 'ivygap'].values
    is_le  = (ivygap == 'LE')
    is_ct  = (ivygap == 'CT')
    if is_le.sum() < 3 or is_ct.sum() < 3:
        continue

    obs_markers = [g for g in MARKER_GENES if g in adata.var_names]
    X = adata[:, obs_markers].X
    X = X.toarray() if issparse(X) else np.asarray(X)
    C_raw = np.round(X).astype(np.int64)

    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    Xn = adata_norm[:, obs_markers].X
    log_cpm = (Xn.toarray() if issparse(Xn) else np.asarray(Xn))

    A      = is_le.astype(int)
    coords = adata.obsm['spatial'].astype(np.float64)
    spot_names = list(adata.obs_names)

    # ── t-test (native diff + SE) ──────────────────────────────────────────────
    for i, g in enumerate(obs_markers):
        le, ct = log_cpm[is_le, i], log_cpm[is_ct, i]
        diff = le.mean() - ct.mean()
        se   = np.sqrt(le.var(ddof=1) / len(le) + ct.var(ddof=1) / len(ct) + 1e-12)
        _, p = stats.ttest_ind(le, ct, equal_var=False)
        records['t-test'].append({'sample': sample, 'gene': g, 'tau': diff, 'se': se, 'pval': p})

    # ── SpaGCN / Wilcoxon (signed z; SE = 1, standardized statistic) ───────────
    for i, g in enumerate(obs_markers):
        le, ct = log_cpm[is_le, i], log_cpm[is_ct, i]
        z, p = stats.ranksums(le, ct)
        records['SpaGCN'].append({'sample': sample, 'gene': g, 'tau': z, 'se': 1.0, 'pval': p})

    # ── DESpace (R subprocess; logFC + SE derived from p) ──────────────────────
    try:
        eff, pv = run_r_method(rscript_despace, C_raw, A, coords, obs_markers, spot_names)
        for i, g in enumerate(obs_markers):
            se = se_from_effect_and_pvalue(eff[i], pv[i])
            if np.isfinite(eff[i]) and np.isfinite(se):
                records['DESpace'].append({'sample': sample, 'gene': g, 'tau': eff[i], 'se': se, 'pval': pv[i]})
    except Exception as e:
        print(f'  [DESpace] {sample} failed: {e}', file=sys.stderr)

    # ── SpatialGEE (R subprocess; GEE coef + SE derived from p) ────────────────
    try:
        eff, pv = run_r_method(rscript_spatialgee, C_raw, A, coords, obs_markers, spot_names)
        for i, g in enumerate(obs_markers):
            se = se_from_effect_and_pvalue(eff[i], pv[i])
            if np.isfinite(eff[i]) and np.isfinite(se):
                records['SpatialGEE'].append({'sample': sample, 'gene': g, 'tau': eff[i], 'se': se, 'pval': pv[i]})
    except Exception as e:
        print(f'  [SpatialGEE] {sample} failed: {e}', file=sys.stderr)


# ── Pool tidest from pre-computed per-sample PLM results ───────────────────────
for _, r in plm_all.iterrows():
    pass  # tidest pooled directly below from plm_all (already has tau, se per sample)


# ── DL-RE pooling per gene, per method ──────────────────────────────────────────
print('\nPooling per-method results across samples with DerSimonian-Laird RE...')
all_genes = [g for g in MARKER_GENES if g in set(plm_all['gene'])]

method_dfs = {}

# tidest
rows = []
for g in all_genes:
    sub = plm_all[plm_all['gene'] == g]
    if len(sub) < 5:
        continue
    tau, se, z, p = dersimonian_laird(sub['tau'].values, sub['se'].values)
    rows.append({'gene': g, 'tau': tau, 'se': se, 'pval': p, 'n_samples': len(sub)})
df_plm_pooled = pd.DataFrame(rows)
df_plm_pooled['qval'] = bh_correct(df_plm_pooled['pval'].values)
method_dfs['tidest'] = df_plm_pooled

# competitors
for name, recs in records.items():
    df_r = pd.DataFrame(recs)
    rows = []
    for g in all_genes:
        sub = df_r[(df_r['gene'] == g) & df_r['se'].notna() & np.isfinite(df_r['se']) & (df_r['se'] > 0)]
        if len(sub) < 5:
            continue
        tau, se, z, p = dersimonian_laird(sub['tau'].values, sub['se'].values)
        rows.append({'gene': g, 'tau': tau, 'se': se, 'pval': p, 'n_samples': len(sub)})
    df_pooled = pd.DataFrame(rows)
    df_pooled['qval'] = bh_correct(df_pooled['pval'].values)
    method_dfs[name] = df_pooled


# ── Score ────────────────────────────────────────────────────────────────────────
def score_results(df):
    df = df.copy()
    df['exp'] = df['gene'].map(EXPECTED)
    known = df[df['exp'].notna() & (df['exp'] != 0)]
    sig_known = known[known['qval'] < ALPHA]
    return {
        'n_genes':       len(df),
        'n_sig':         int((df['qval'] < ALPHA).sum()),
        'n_known':       len(known),
        'n_sig_known':   len(sig_known),
        'n_sig_correct': int((sig_known['tau'] * sig_known['exp'] > 0).sum()),
        'dir_acc_all':   float((known['tau'] * known['exp'] > 0).mean()) if len(known) else np.nan,
        'dir_acc_sig':   float((sig_known['tau'] * sig_known['exp'] > 0).mean()) if len(sig_known) else np.nan,
    }

print('\n── Scoring (pooled via DL-RE across per-sample fits) ───────────────────────')
score_rows = []
for name, df in method_dfs.items():
    s = score_results(df)
    s['method'] = name
    score_rows.append(s)
    print(f'{name:<12} n={s["n_genes"]:2d}  sig={s["n_sig"]:2d}  sig_known={s["n_sig_known"]:2d}  '
          f'sig_correct={s["n_sig_correct"]:2d}  dir_all={s["dir_acc_all"]:.1%}  dir_sig={s["dir_acc_sig"]:.1%}')

score_df = pd.DataFrame(score_rows).set_index('method')
score_df.to_csv(OUT_SUMMARY)
print(f'\nSaved {OUT_SUMMARY}')

# ── Per-gene wide table ──────────────────────────────────────────────────────────
wide = pd.DataFrame({'gene': all_genes}).set_index('gene')
for name, df in method_dfs.items():
    key = name.lower().replace('-', '').replace(' ', '')
    df_i = df.set_index('gene')
    wide[f'tau_{key}']  = df_i['tau']
    wide[f'se_{key}']   = df_i['se']
    wide[f'pval_{key}'] = df_i['pval']
    wide[f'qval_{key}'] = df_i['qval']
wide.to_csv(OUT_PERGENE)
print(f'Saved {OUT_PERGENE}')
print('Done.')
