"""
Rounds 1-3 core analysis: Spatial error model, spatial lag model,
ground truth alternatives, cross-model validation.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings; warnings.filterwarnings('ignore')
import json, os

BASE = "/Users/dingyizhuang/MIT Dropbox/Dingyi Zhuang/geo-llm-bias"

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(f"{BASE}/data/processed/analysis_dataset.csv")
gpt = pd.read_csv(f"{BASE}/data/processed/llm_responses_gpt.csv")

qol_en = gpt[gpt['query_type']=='qol_en'][['geonameid','numeric_response']].rename(columns={'numeric_response':'llm_qol_en'})
qol_zh = gpt[gpt['query_type']=='qol_zh'][['geonameid','numeric_response']].rename(columns={'numeric_response':'llm_qol_zh'})
df = df.merge(qol_en, on='geonameid', how='left')
df = df.merge(qol_zh, on='geonameid', how='left')

def norm(s):
    mn, mx = s.min(), s.max()
    return (s-mn)/(mx-mn)*9+1 if mx > mn else pd.Series([5.5]*len(s), index=s.index)

df['gt_qol']   = df.groupby('continent')['gdp_per_capita'].transform(norm)
df['bias_en']  = df['llm_qol_en'] - df['gt_qol']
df['bias_zh']  = df['llm_qol_zh'] - df['gt_qol']
df['bias_claude'] = df['claude_qol_en'] - df['gt_qol']
df = df[df['continent'] != 'Other'].dropna(subset=['bias_en','gdp_per_capita','internet_pct','latitude','longitude'])
print(f"N after filter: {len(df)}")

results = {}

# ── 1. Spatial Error Model (PRIMARY spec) ───────────────────────────────────
print("\n=== SPATIAL ERROR MODEL (primary specification) ===")
try:
    import geopandas as gpd
    from libpysal.weights import KNN
    import spreg

    gdf = gpd.GeoDataFrame(df.reset_index(drop=True),
                           geometry=gpd.points_from_xy(df['longitude'], df['latitude'])).set_crs('EPSG:4326')
    w5 = KNN.from_dataframe(gdf, k=5)
    w5.transform = 'r'

    X_cols = ['log_gdp_pc','internet_pct','log_wiki','log_pop']
    # Continent dummies
    df_r = df.reset_index(drop=True)
    for c in ['Asia','Europe','North America','South America','Oceania']:
        df_r[f'cont_{c.replace(" ","_")}'] = (df_r['continent']==c).astype(float)
    cont_cols = [f'cont_{c.replace(" ","_")}' for c in ['Asia','Europe','North America','South America','Oceania']]
    all_X = X_cols + cont_cols

    df_model = df_r.dropna(subset=['bias_en']+all_X).reset_index(drop=True)
    # Realign weights to this subset
    idx = df_model.index.tolist()
    gdf_sub = gpd.GeoDataFrame(df_model, geometry=gpd.points_from_xy(df_model['longitude'], df_model['latitude'])).set_crs('EPSG:4326')
    w_sub = KNN.from_dataframe(gdf_sub, k=5)
    w_sub.transform = 'r'

    y = df_model['bias_en'].values
    X = df_model[all_X].values

    # Spatial Error Model
    sem = spreg.GM_Error(y, X, w_sub, name_y='bias_en', name_x=all_X)
    print(f"  SEM lambda={sem.lam:.4f}, pseudo-R²≈{1-sem.sig2/np.var(y):.3f}")

    # Spatial Lag Model
    slm = spreg.GM_Lag(y, X, w_sub, name_y='bias_en', name_x=all_X)
    print(f"  SLM rho={slm.rho:.4f}")

    # OLS for comparison
    ols = spreg.OLS(y, X, name_y='bias_en', name_x=all_X)
    print(f"  OLS R²={ols.r2:.4f}")

    # Extract SEM coefficients
    sem_coefs = {}
    for i, nm in enumerate(['Intercept']+all_X):
        sem_coefs[nm] = {'coef': float(sem.betas[i][0]), 'std_err': float(sem.std_err[i])}
    sem_coefs['lambda'] = float(sem.lam)

    results['spatial_error_model'] = {
        'lambda': float(sem.lam),
        'gdp_coef': float(sem.betas[1][0]),
        'gdp_se': float(sem.std_err[1]),
        'gdp_z': float(sem.z_stat[1][0]),
        'gdp_p': float(sem.z_stat[1][1]),
        'pseudo_R2': float(1 - sem.sig2/np.var(y)),
        'n': int(len(y)),
        'all_coefs': sem_coefs,
    }
    results['spatial_lag_model'] = {
        'rho': float(slm.rho),
        'gdp_coef': float(slm.betas[1][0]),
        'n': int(len(y)),
    }
    results['ols_baseline'] = {'R2': float(ols.r2), 'n': int(len(y))}
    print(f"\n  SEM: GDP β={sem.betas[1][0]:.4f}, SE={sem.std_err[1]:.4f}, z={sem.z_stat[1][0]:.2f}, p={sem.z_stat[1][1]:.4f}")

except Exception as e:
    print(f"  Spatial model error: {e}")
    import traceback; traceback.print_exc()

# ── 2. Ground Truth Alternatives ─────────────────────────────────────────────
print("\n=== GROUND TRUTH SENSITIVITY ===")
gt_results = {}

# GT1: Within-continent GDP (baseline)
df['gt_1_gdp_cont'] = df.groupby('continent')['gdp_per_capita'].transform(norm)
df['bias_gt1'] = df['llm_qol_en'] - df['gt_1_gdp_cont']

# GT2: Global GDP normalization
df['gt_2_gdp_global'] = norm(df['gdp_per_capita'])
df['bias_gt2'] = df['llm_qol_en'] - df['gt_2_gdp_global']

# GT3: Population-weighted proxy (larger cities tend higher QoL score assignment)
df['gt_3_pop'] = df.groupby('continent')['population'].transform(norm)
df['bias_gt3'] = df['llm_qol_en'] - df['gt_3_pop']

# GT4: Internet + GDP composite
df['gdp_internet_composite'] = 0.7*df['gdp_per_capita'] + 0.3*(df['internet_pct']/100 * df['gdp_per_capita'].max())
df['gt_4_composite'] = df.groupby('continent')['gdp_internet_composite'].transform(norm)
df['bias_gt4'] = df['llm_qol_en'] - df['gt_4_composite']

for gt_name, bias_col in [('Within-continent GDP', 'bias_gt1'),
                            ('Global GDP', 'bias_gt2'),
                            ('Population proxy', 'bias_gt3'),
                            ('GDP+Internet composite', 'bias_gt4')]:
    m = smf.ols(f'{bias_col} ~ log_gdp_pc + internet_pct + log_wiki + log_pop + C(continent)',
                data=df.dropna(subset=[bias_col,'log_gdp_pc','internet_pct'])).fit(cov_type='HC3')
    corr_gdp = df[[bias_col,'gdp_per_capita']].dropna().corr().iloc[0,1]
    gt_results[gt_name] = {'R2': round(m.rsquared,3), 'gdp_coef': round(float(m.params['log_gdp_pc']),3),
                            'gdp_p': round(float(m.pvalues['log_gdp_pc']),4),
                            'bias_gdp_corr': round(corr_gdp,3)}
    print(f"  {gt_name}: R²={m.rsquared:.3f}, β_GDP={m.params['log_gdp_pc']:.3f}, corr(bias,GDP)={corr_gdp:.3f}")

results['ground_truth_sensitivity'] = gt_results

# GT consistency check: do all GTs agree on direction?
print("\n  Cross-GT bias correlation matrix:")
bias_cols = ['bias_gt1','bias_gt2','bias_gt3','bias_gt4']
corr_mat = df[bias_cols].corr().round(3)
print(corr_mat)
results['gt_cross_correlation'] = corr_mat.to_dict()

# ── 3. Cross-model detailed ──────────────────────────────────────────────────
print("\n=== CROSS-MODEL DETAILED COMPARISON ===")
df_cm = df.dropna(subset=['bias_en','bias_claude'])

from scipy import stats
corr_r, corr_p = stats.pearsonr(df_cm['bias_en'], df_cm['bias_claude'])
wilcox = stats.wilcoxon(df_cm['bias_en'], df_cm['bias_claude'])
print(f"  Pearson r={corr_r:.4f} (p={corr_p:.4f})")
print(f"  Wilcoxon p={wilcox.pvalue:.4f}")
print(f"  GPT mean bias: {df_cm['bias_en'].mean():.3f}")
print(f"  Claude mean bias: {df_cm['bias_claude'].mean():.3f}")
print(f"  Mean absolute difference: {(df_cm['bias_en']-df_cm['bias_claude']).abs().mean():.3f}")

# Continent breakdown
for cont, grp in df_cm.groupby('continent'):
    r, p = stats.pearsonr(grp['bias_en'], grp['bias_claude'])
    print(f"    {cont}: r={r:.3f}, p={p:.3f}, n={len(grp)}")

results['cross_model'] = {
    'pearson_r': round(corr_r, 4), 'pearson_p': round(corr_p, 4),
    'wilcoxon_p': round(wilcox.pvalue, 4),
    'gpt_mean': round(df_cm['bias_en'].mean(), 3),
    'claude_mean': round(df_cm['bias_claude'].mean(), 3),
    'mean_abs_diff': round((df_cm['bias_en']-df_cm['bias_claude']).abs().mean(), 3),
}

# ── 4. Spatial lag model residuals diagnostics ───────────────────────────────
print("\n=== OLS RESIDUAL DIAGNOSTICS ===")
m_full = smf.ols('bias_en ~ log_gdp_pc + internet_pct + log_wiki + log_pop + C(continent)',
                  data=df.dropna(subset=['bias_en','log_gdp_pc','internet_pct'])).fit(cov_type='HC3')
df_resid = df.dropna(subset=['bias_en','log_gdp_pc','internet_pct']).copy()
df_resid['resid'] = m_full.resid
print(f"  OLS R²={m_full.rsquared:.4f}, N={int(m_full.nobs)}")
print(f"  Residual mean: {df_resid['resid'].mean():.4f}, std: {df_resid['resid'].std():.4f}")

# ── Save ────────────────────────────────────────────────────────────────────
with open(f"{BASE}/data/processed/spatial_analysis.json", 'w') as f:
    json.dump(results, f, indent=2, default=str)

df_resid[['geonameid','name','continent','bias_en','resid','gdp_per_capita','latitude','longitude']].to_csv(
    f"{BASE}/data/processed/ols_residuals.csv", index=False)

print("\n=== SPATIAL ANALYSIS COMPLETE ===")
print(json.dumps({k: v for k,v in results.items() if k != 'all_coefs'}, indent=2, default=str)[:2000])
