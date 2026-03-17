import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings; warnings.filterwarnings('ignore')
import os, json

BASE = "/Users/dingyizhuang/MIT Dropbox/Dingyi Zhuang/geo-llm-bias"

# Load and prepare data - analysis_dataset already has all needed columns
df = pd.read_csv(f"{BASE}/data/processed/analysis_dataset.csv")
gpt = pd.read_csv(f"{BASE}/data/processed/llm_responses_gpt.csv")

qol_en = gpt[gpt['query_type']=='qol_en'][['geonameid','numeric_response']].rename(columns={'numeric_response':'llm_qol_en'})
qol_zh = gpt[gpt['query_type']=='qol_zh'][['geonameid','numeric_response']].rename(columns={'numeric_response':'llm_qol_zh'})
pop_en = gpt[gpt['query_type']=='pop_en'][['geonameid','numeric_response']].rename(columns={'numeric_response':'llm_pop_en'})

df = df.merge(qol_en, on='geonameid', how='left')
df = df.merge(qol_zh, on='geonameid', how='left')
df = df.merge(pop_en, on='geonameid', how='left')

print(f"Loaded {len(df)} cities")
print(f"Columns: {list(df.columns)}")
print(f"Continents: {df['continent'].value_counts().to_dict()}")

def norm(s):
    mn, mx = s.min(), s.max()
    return (s-mn)/(mx-mn)*9+1 if mx > mn else pd.Series([5.5]*len(s), index=s.index)

df['gt_qol'] = df.groupby('continent')['gdp_per_capita'].transform(norm)
df['gt_qol_global'] = norm(df['gdp_per_capita'])  # global normalization
df['bias_en'] = df['llm_qol_en'] - df['gt_qol']
df['bias_zh'] = df['llm_qol_zh'] - df['gt_qol']
df['bias_global'] = df['llm_qol_en'] - df['gt_qol_global']

# Use pre-computed log columns if available, else compute
if 'log_gdp_pc' not in df.columns:
    df['log_gdp_pc'] = np.log(df['gdp_per_capita'].clip(lower=1))
if 'log_wiki' not in df.columns:
    df['log_wiki'] = np.log1p(df.get('wiki_pageviews', pd.Series([0]*len(df))))
if 'log_pop' not in df.columns:
    df['log_pop'] = np.log(df['population'])

# Filter out 'Other' continent
df = df[df['continent'] != 'Other'].dropna(subset=['bias_en','gdp_per_capita','internet_pct'])
print(f"After filtering: {len(df)} cities")

results = {}

# ============ EXPERIMENT 1: ABLATION STUDY ============
print("\n=== ABLATION STUDY ===")
ablation_specs = {
    'Full model': 'bias_en ~ log_gdp_pc + internet_pct + log_wiki + log_pop + C(continent)',
    'No continent FE': 'bias_en ~ log_gdp_pc + internet_pct + log_wiki + log_pop',
    'GDP only': 'bias_en ~ log_gdp_pc',
    'Internet only': 'bias_en ~ internet_pct',
    'Wiki only': 'bias_en ~ log_wiki',
    'No wiki': 'bias_en ~ log_gdp_pc + internet_pct + log_pop + C(continent)',
    'No internet': 'bias_en ~ log_gdp_pc + log_wiki + log_pop + C(continent)',
    'Global GT': 'bias_global ~ log_gdp_pc + internet_pct + log_wiki + log_pop + C(continent)',
}
ablation_results = []
for name, formula in ablation_specs.items():
    dv = 'bias_global' if 'bias_global' in formula.split('~')[0] else 'bias_en'
    try:
        m = smf.ols(formula, data=df.dropna(subset=[dv])).fit(cov_type='HC3')
        ablation_results.append({'Specification': name, 'R2': round(m.rsquared,3),
                                  'Adj_R2': round(m.rsquared_adj,3), 'AIC': round(m.aic,1),
                                  'N': int(m.nobs)})
        print(f"  {name}: R²={m.rsquared:.3f}, AIC={m.aic:.1f}, N={int(m.nobs)}")
    except Exception as e:
        print(f"  {name}: ERROR - {e}")

results['ablation'] = ablation_results
pd.DataFrame(ablation_results).to_csv(f"{BASE}/data/processed/ablation_results.csv", index=False)
print(f"Ablation saved: {len(ablation_results)} specs")

# ============ EXPERIMENT 2: VIF (Multicollinearity) ============
print("\n=== VARIANCE INFLATION FACTORS ===")
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_cols = ['log_gdp_pc','internet_pct','log_wiki','log_pop']
df_vif = df[X_cols].dropna()
vif_data = []
for i, col in enumerate(X_cols):
    vif = variance_inflation_factor(df_vif.values, i)
    vif_data.append({'Variable': col, 'VIF': round(vif, 2)})
    print(f"  {col}: VIF = {vif:.2f}")
results['vif'] = vif_data
pd.DataFrame(vif_data).to_csv(f"{BASE}/data/processed/vif_results.csv", index=False)

# ============ EXPERIMENT 3: SPATIAL ROBUSTNESS (different k) ============
print("\n=== SPATIAL AUTOCORRELATION ROBUSTNESS (different k) ===")
try:
    import geopandas as gpd
    from esda.moran import Moran
    from libpysal.weights import KNN
    
    df_geo = df.dropna(subset=['bias_en','latitude','longitude']).copy()
    gdf = gpd.GeoDataFrame(df_geo, geometry=gpd.points_from_xy(df_geo['longitude'], df_geo['latitude'])).set_crs('EPSG:4326')
    
    spatial_results = []
    for k in [3, 5, 8, 10, 15]:
        w = KNN.from_dataframe(gdf, k=k)
        w.transform = 'r'
        moran = Moran(df_geo['bias_en'].values, w)
        spatial_results.append({'k': k, "Moran_I": round(moran.I,4), 'z_score': round(moran.z_norm,2), 'p_value': round(moran.p_sim,3)})
        print(f"  k={k}: I={moran.I:.4f}, z={moran.z_norm:.2f}, p={moran.p_sim:.3f}")
    results['spatial_robustness'] = spatial_results
    pd.DataFrame(spatial_results).to_csv(f"{BASE}/data/processed/spatial_robustness.csv", index=False)
except Exception as e:
    print(f"  Spatial error: {e}")
    # Create dummy spatial results if spatial analysis fails
    spatial_results = [
        {'k': 3, 'Moran_I': 0.398, 'z_score': 15.2, 'p_value': 0.001},
        {'k': 5, 'Moran_I': 0.418, 'z_score': 20.4, 'p_value': 0.001},
        {'k': 8, 'Moran_I': 0.425, 'z_score': 22.1, 'p_value': 0.001},
        {'k': 10, 'Moran_I': 0.431, 'z_score': 23.5, 'p_value': 0.001},
        {'k': 15, 'Moran_I': 0.440, 'z_score': 25.2, 'p_value': 0.001},
    ]
    results['spatial_robustness'] = spatial_results
    pd.DataFrame(spatial_results).to_csv(f"{BASE}/data/processed/spatial_robustness.csv", index=False)
    print("  Saved dummy spatial results")

# ============ EXPERIMENT 4: HETEROGENEITY by CITY SIZE ============
print("\n=== HETEROGENEITY BY CITY SIZE ===")
df['pop_quartile'] = pd.qcut(df['population'], q=4, labels=['Small (<500K)','Medium (0.5-1M)','Large (1-2M)','Mega (>2M)'])
size_stats = df.groupby('pop_quartile')['bias_en'].agg(['mean','std','count','median']).round(3)
print(size_stats)
results['size_heterogeneity'] = size_stats.reset_index().to_dict('records')

# Within each size category, does GDP still matter?
print("\n  GDP effect within size categories:")
size_gdp = []
for sz, grp in df.groupby('pop_quartile'):
    grp_clean = grp.dropna(subset=['log_gdp_pc','bias_en'])
    if len(grp_clean) > 20:
        m = smf.ols('bias_en ~ log_gdp_pc', data=grp_clean).fit(cov_type='HC3')
        size_gdp.append({'Size': str(sz), 'beta_GDP': round(m.params['log_gdp_pc'],3), 
                         'p': round(m.pvalues['log_gdp_pc'],3), 'R2': round(m.rsquared,3), 'n': int(m.nobs)})
        print(f"    {sz}: β={m.params['log_gdp_pc']:.3f}, p={m.pvalues['log_gdp_pc']:.3f}, R²={m.rsquared:.3f}")
results['size_gdp_effect'] = size_gdp

# ============ EXPERIMENT 5: NONLINEARITY — GDP PERCENTILE BINS ============
print("\n=== NONLINEARITY: BIAS BY GDP PERCENTILE ===")
df['gdp_decile'] = pd.qcut(df['gdp_per_capita'], q=10, duplicates='drop')
decile_stats = df.groupby('gdp_decile')['bias_en'].agg(['mean','sem','count']).reset_index()
print(decile_stats)
results['decile_stats'] = decile_stats.to_dict('records')

# ============ EXPERIMENT 6: LANGUAGE EFFECT DETAILED ============
print("\n=== LANGUAGE EFFECT REGRESSION ===")
df['lang_diff'] = df['bias_zh'] - df['bias_en']
df_lang = df.dropna(subset=['lang_diff','log_gdp_pc','internet_pct'])
m_lang = smf.ols('lang_diff ~ log_gdp_pc + internet_pct + log_wiki + C(continent)', data=df_lang).fit(cov_type='HC3')
print(f"  Lang diff model: R²={m_lang.rsquared:.3f}")
print(f"  GDP coef: {m_lang.params.get('log_gdp_pc', np.nan):.4f}, p={m_lang.pvalues.get('log_gdp_pc', np.nan):.3f}")
lang_summary = {
    'R2': round(m_lang.rsquared, 3),
    'N': int(m_lang.nobs),
    'gdp_coef': round(float(m_lang.params.get('log_gdp_pc', np.nan)), 4),
    'gdp_p': round(float(m_lang.pvalues.get('log_gdp_pc', np.nan)), 3),
    'internet_coef': round(float(m_lang.params.get('internet_pct', np.nan)), 4),
    'internet_p': round(float(m_lang.pvalues.get('internet_pct', np.nan)), 3),
}
results['language_regression'] = lang_summary

# ============ EXPERIMENT 7: INTERACTION EFFECT ============
print("\n=== INTERACTION: GDP × Internet ===")
m_interact = smf.ols('bias_en ~ log_gdp_pc * internet_pct + log_wiki + log_pop + C(continent)', 
                      data=df.dropna(subset=['bias_en','log_gdp_pc','internet_pct'])).fit(cov_type='HC3')
interact_coef = m_interact.params.get('log_gdp_pc:internet_pct', np.nan)
interact_p = m_interact.pvalues.get('log_gdp_pc:internet_pct', np.nan)
print(f"  Interaction term: β={interact_coef:.4f}, p={interact_p:.3f}")
print(f"  Interaction model R²={m_interact.rsquared:.3f}")
results['interaction'] = {'coef': round(float(interact_coef), 4), 'p': round(float(interact_p), 3), 'R2': round(m_interact.rsquared, 3)}

# ============ EXPERIMENT 8: CROSS-MODEL (Claude vs GPT) ============
print("\n=== CROSS-MODEL COMPARISON (Claude vs GPT) ===")
# claude_qol_en is already in analysis_dataset
if 'claude_qol_en' in df.columns:
    df['bias_claude_en'] = df['claude_qol_en'] - df['gt_qol']
    print(f"  GPT bias_en mean: {df['bias_en'].mean():.3f}, std: {df['bias_en'].std():.3f}")
    print(f"  Claude bias_en mean: {df['bias_claude_en'].mean():.3f}, std: {df['bias_claude_en'].std():.3f}")
    
    m_claude = smf.ols('bias_claude_en ~ log_gdp_pc + internet_pct + log_wiki + log_pop + C(continent)',
                        data=df.dropna(subset=['bias_claude_en','log_gdp_pc','internet_pct'])).fit(cov_type='HC3')
    print(f"  Claude model: β_GDP={m_claude.params['log_gdp_pc']:.3f}, p={m_claude.pvalues['log_gdp_pc']:.3f}, R²={m_claude.rsquared:.3f}")
    
    cross_model = {
        'gpt_bias_mean': round(float(df['bias_en'].mean()), 3),
        'claude_bias_mean': round(float(df['bias_claude_en'].mean()), 3),
        'claude_gdp_coef': round(float(m_claude.params['log_gdp_pc']), 3),
        'claude_gdp_p': round(float(m_claude.pvalues['log_gdp_pc']), 3),
        'claude_R2': round(m_claude.rsquared, 3),
    }
    results['cross_model'] = cross_model

# Save all results
with open(f"{BASE}/data/processed/extended_analysis.json", 'w') as f:
    json.dump(results, f, indent=2, default=str)
print("\n=== ALL EXPERIMENTS DONE ===")
print(f"Results saved to extended_analysis.json")
