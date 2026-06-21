#!/usr/bin/env python3
"""Adversarial independent re-derivation of the 4 reanalysis headline numbers.
Builds domain scores from raw artifacts using the validated plumbing, then
recomputes C4 hierarchical R2 / partial r, scaling LMM, signature binomials,
and scorer separability baseline INDEPENDENTLY."""
import json, glob, csv
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, binomtest
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd

STATIC_DIR = "results/full_eval_20260526_2208"
MT_DIR = "results/multiturn_eval_v3"

DOMAIN_MAP = {
    'n_back': 'WM', 'digit_span': 'WM', 'operation_span': 'WM',
    'stroop': 'Control', 'flanker': 'Control', 'go_nogo': 'Control',
    'cvlt_word_list': 'Episodic', 'drm_false_memory': 'Episodic', 'source_monitoring': 'Episodic',
    'false_belief': 'ToM', 'epitome_tom': 'ToM',
    'confidence_calibration': 'Meta', 'post_decision_wagering': 'Meta'
}
SIZE_MAP = {
    'tinyllama:1.1b': 1.1, 'qwen2.5:0.5b': 0.5, 'qwen2.5:1.5b': 1.5,
    'gemma2:2b': 2, 'llama3.2:1b': 1, 'qwen2.5:3b': 3, 'llama3.2:3b': 3,
    'qwen2.5:7b': 7, 'mistral:7b': 7, 'llama3.1:8b': 8, 'deepseek-r1:7b': 7,
    'gemma2:9b': 9, 'qwen2.5:14b': 14, 'phi3:14b': 14, 'deepseek-r1:14b': 14,
    'gemma2:27b': 27, 'qwen2.5:32b': 32, 'mixtral:8x7b': 47, 'yi:34b': 34,
    'command-r:35b': 35
}
FAMILY = {m: ('qwen2.5' if m.startswith('qwen2.5') else
             'gemma2' if m.startswith('gemma2') else
             'llama3.2' if m.startswith('llama3.2') else
             'llama3.1' if m.startswith('llama3.1') else
             'deepseek-r1' if m.startswith('deepseek-r1') else
             m.split(':')[0]) for m in SIZE_MAP}

def build_paradigm_matrix():
    static = {}
    for f in sorted(glob.glob(f"{STATIC_DIR}/*/text/aggregate.json")):
        d = json.load(open(f))
        m = d["model"].replace("openai/", "")
        static[m] = {p: v["accuracy"] for p, v in d["paradigms"].items()}
    mt = {}
    for f in sorted(glob.glob(f"{MT_DIR}/*/aggregate.json")):
        d = json.load(open(f))
        m = d["model"].replace("openai/", "")
        mt[m] = {k.split("/")[-1]: v["accuracy"] for k, v in d.get("dimensions", {}).items()}
    merged = {}
    for m in static:
        merged[m] = dict(static[m])
        if m in mt:
            for p in ["n_back", "operation_span", "cvlt_word_list"]:
                if p in mt[m]:
                    merged[m][p] = mt[m][p]
    return merged

merged = build_paradigm_matrix()
# Keep only the 20 cogarena models present in SIZE_MAP
models = [m for m in SIZE_MAP if m in merged]
print("N paradigm-matrix models:", len(models))

# Domain scores (mean of paradigms in domain)
domain_scores = {}
for m in models:
    ds = {}
    for dom in ["WM", "Control", "Episodic", "ToM", "Meta"]:
        ps = [merged[m][p] for p in merged[m] if DOMAIN_MAP.get(p) == dom]
        ds[dom] = float(np.mean(ps)) if ps else 0.0
    domain_scores[m] = ds

# ===== C4 (REMOVED) =====
# The C4 agent-battery analysis (former paper Section 5.7) was cut from the paper
# (see memory wmfam-agent-battery-cut); its verification block and input
# data/wmfam_overlap.csv were archived to results/_archive/orphaned_2026-06-21/.
# The live verifications below (scaling LMM, signatures, separability baseline)
# are self-contained and unaffected.

# ===== SCALING (LMM spot-check the 4 strong paradigms + 2 near-zero) =====
print("\n===== SCALING LMM =====")
def lmm_slope(paradigm):
    rows=[]
    for m in models:
        if paradigm in merged[m]:
            rows.append({'acc':merged[m][paradigm],'logsize':np.log10(SIZE_MAP[m]),'family':FAMILY[m]})
    df=pd.DataFrame(rows)
    md = smf.mixedlm("acc ~ logsize", df, groups=df["family"], re_formula="~1")
    res = md.fit(reml=True, method='lbfgs')
    fe = res.fe_params['logsize']; bse = res.bse['logsize']
    ci=(fe-1.96*bse, fe+1.96*bse)
    p = res.pvalues['logsize']
    conv = res.converged
    # naive pearson
    x=df['logsize'].values; yv=df['acc'].values
    pr,pp=pearsonr(x,yv)
    return fe,bse,ci,p,conv,pr,pp

for par in ['stroop','epitome_tom','drm_false_memory','false_belief','n_back','cvlt_word_list','go_nogo']:
    fe,bse,ci,p,conv,pr,pp=lmm_slope(par)
    print(f"{par}: lmm_slope={fe:.4f} CI=[{ci[0]:.4f},{ci[1]:.4f}] p={p:.3e} conv={conv} | naive_r={pr:.4f} p={pp:.4f}")

# ===== SIGNATURE binomial spot-check =====
print("\n===== SIGNATURE binomial =====")
# Re-derive p from k/n using one-sided binom (greater), H0=0.5
from statsmodels.stats.multitest import multipletests
sig = {'flanker':17,'n_back_load':16,'epitome':15,'false_belief':12,'stroop':9}
raw_ps={}
for par,k in sig.items():
    bt=binomtest(k,20,0.5,alternative='greater')
    raw_ps[par]=bt.pvalue
    print(f"{par}: k={k}/20 one-sided binom greater p={bt.pvalue:.6f}")
# BH correction across 5
names=list(raw_ps.keys())
pv=[raw_ps[n] for n in names]
rej,padj,_,_=multipletests(pv,alpha=0.05,method='fdr_bh')
print("BH adjusted:")
for n,pa,rj in zip(names,padj,rej):
    print(f"  {n}: p_bh={pa:.6f} sig={rj}")

# ===== SCORER separability baseline =====
print("\n===== SCORER baseline separability =====")
# Build 13-paradigm matrix (generic), compute within-minus-cross delta with 5000-perm test seed=42
PARS=list(DOMAIN_MAP.keys())
X = np.array([[merged[m][p] for p in PARS] for m in models])  # 20x13
# correlation across models between paradigms (columns)
C = np.corrcoef(X.T)  # 13x13
dom_of=[DOMAIN_MAP[p] for p in PARS]
within=[]; cross=[]
for i in range(13):
    for j in range(i+1,13):
        if dom_of[i]==dom_of[j]: within.append(C[i,j])
        else: cross.append(C[i,j])
wm=np.mean(within); cm=np.mean(cross)
print(f"within_mean={wm:.4f} cross_mean={cm:.4f} delta={wm-cm:.4f} (n_within={len(within)} n_cross={len(cross)})")
# permutation test
rng=np.random.default_rng(42)
obs=wm-cm
perm=[]
idx_pairs=[(i,j) for i in range(13) for j in range(i+1,13)]
labels=np.array(dom_of)
for _ in range(5000):
    pl=rng.permutation(labels)
    w=[];c=[]
    for (i,j) in idx_pairs:
        if pl[i]==pl[j]: w.append(C[i,j])
        else: c.append(C[i,j])
    perm.append(np.mean(w)-np.mean(c))
perm=np.array(perm)
pval=(np.sum(perm>=obs)+1)/(len(perm)+1)
print(f"perm p (>= obs)={pval:.4f}")
