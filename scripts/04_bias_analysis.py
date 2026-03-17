"""
Step 4-5: Bias scoring, regression analysis, and spatial autocorrelation
"""
import os, warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
warnings.filterwarnings('ignore')

BASE = "/Users/dingyizhuang/MIT Dropbox/Dingyi Zhuang/geo-llm-bias"

# ─── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
cities = pd.read_csv(f"{BASE}/data/raw/cities_sample.csv")
gdp    = pd.read_csv(f"{BASE}/data/raw/worldbank_gdp.csv")
net    = pd.read_csv(f"{BASE}/data/raw/worldbank_internet.csv")
wiki   = pd.read_csv(f"{BASE}/data/raw/wikipedia_pageviews.csv")

gpt_resp   = pd.read_csv(f"{BASE}/data/processed/llm_responses_gpt.csv")
claude_resp= pd.read_csv(f"{BASE}/data/processed/llm_responses_claude.csv")

# Detect synthetic or real
is_synthetic = 'synthetic' in gpt_resp['model'].iloc[0].lower()
print(f"Data type: {'SYNTHETIC' if is_synthetic else 'REAL'}")

# ─── Build analysis dataset ────────────────────────────────────────────────────
# Merge covariates
df = cities[['geonameid','name','asciiname','latitude','longitude','country_code',
             'population','continent']].copy()
df = df.merge(gdp, on='country_code', how='left')
df = df.merge(net, on='country_code', how='left')
df = df.merge(wiki[['geonameid','wiki_pageviews']], on='geonameid', how='left')

# Wide-format LLM scores: pivot to one row per city
def pivot_responses(resp_df, prefix):
    # QoL English
    qol_en = resp_df[resp_df['query_type']=='qol_en'][['geonameid','numeric_response']].copy()
    qol_en.columns = ['geonameid', f'{prefix}_qol_en']
    # QoL Chinese
    qol_zh = resp_df[resp_df['query_type']=='qol_zh'][['geonameid','numeric_response']].copy()
    qol_zh.columns = ['geonameid', f'{prefix}_qol_zh']
    # Population estimate
    pop_en = resp_df[resp_df['query_type']=='pop_en'][['geonameid','numeric_response']].copy()
    pop_en.columns = ['geonameid', f'{prefix}_pop_en']
    return qol_en.merge(qol_zh, on='geonameid', how='outer').merge(pop_en, on='geonameid', how='outer')

gpt_wide   = pivot_responses(gpt_resp,   'gpt')
claude_wide= pivot_responses(claude_resp,'claude')

df = df.merge(gpt_wide,    on='geonameid', how='left')
df = df.merge(claude_wide, on='geonameid', how='left')

# ─── Feature engineering ──────────────────────────────────────────────────────
df['log_gdp_pc']  = np.log1p(df['gdp_per_capita'].fillna(df['gdp_per_capita'].median()))
df['log_wiki']    = np.log1p(df['wiki_pageviews'].fillna(0))
df['log_pop']     = np.log1p(df['population'])
df['internet_pct']= df['internet_pct'].fillna(df['internet_pct'].median())

# Bias score = LLM QoL rating (higher = more positive bias)
# Population bias = log(LLM estimate / actual population)
pop_actual_m = df['population'] / 1e6  # in millions
df['gpt_pop_bias']   = np.log(df['gpt_pop_en'].clip(0.01)   / pop_actual_m.clip(0.01))
df['claude_pop_bias']= np.log(df['claude_pop_en'].clip(0.01) / pop_actual_m.clip(0.01))

# Lang bias (English - Chinese) for QoL
df['gpt_lang_bias']   = df['gpt_qol_en']   - df['gpt_qol_zh']
df['claude_lang_bias']= df['claude_qol_en'] - df['claude_qol_zh']

print(f"\nDataset shape: {df.shape}")
print(f"Cities with GPT QoL: {df['gpt_qol_en'].notna().sum()}")
print(f"Cities with Claude QoL: {df['claude_qol_en'].notna().sum()}")
print(f"\nDescriptive stats for GPT QoL (EN):")
print(df['gpt_qol_en'].describe())
print(f"\nDescriptive stats by continent (GPT QoL EN):")
print(df.groupby('continent')['gpt_qol_en'].agg(['mean','std','count']))

df.to_csv(f"{BASE}/data/processed/analysis_dataset.csv", index=False)
print(f"\nSaved analysis_dataset.csv")

# ─── Regression Analysis ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("REGRESSION ANALYSIS: GPT QoL Bias")
print("="*60)

reg_df = df.dropna(subset=['gpt_qol_en','log_gdp_pc','internet_pct','log_wiki','log_pop'])
# Drop single-value continents
continent_counts = reg_df['continent'].value_counts()
valid_continents = continent_counts[continent_counts >= 3].index
reg_df = reg_df[reg_df['continent'].isin(valid_continents)].copy()

print(f"Regression N = {len(reg_df)}")

# OLS with continent FEs
formula = "gpt_qol_en ~ log_gdp_pc + internet_pct + log_wiki + log_pop + C(continent)"
model = smf.ols(formula, data=reg_df).fit(cov_type='HC3')
print(model.summary())

# Save regression summary
with open(f"{BASE}/data/processed/regression_results.txt", 'w') as f:
    f.write("GPT-4o-mini QoL Bias Regression (HC3 Robust SE)\n")
    f.write("="*60 + "\n")
    f.write(str(model.summary()))
    f.write("\n\n")

# Extract key results
params = model.params
pvalues = model.pvalues
bse = model.bse

print("\n--- Key Coefficients ---")
key_vars = ['log_gdp_pc', 'internet_pct', 'log_wiki', 'log_pop']
results_data = []
for var in key_vars:
    if var in params:
        print(f"{var}: coef={params[var]:.4f}, SE={bse[var]:.4f}, p={pvalues[var]:.4f}")
        results_data.append({
            'variable': var,
            'coefficient': params[var],
            'std_error': bse[var],
            'p_value': pvalues[var],
            'significant': pvalues[var] < 0.05
        })

res_df = pd.DataFrame(results_data)
res_df.to_csv(f"{BASE}/data/processed/regression_coefficients.csv", index=False)

# Find top predictor
sig_res = res_df[res_df['significant']].copy()
if len(sig_res) > 0:
    top_pred = sig_res.reindex(sig_res['coefficient'].abs().sort_values(ascending=False).index).iloc[0]
    print(f"\nTop predictor: {top_pred['variable']}")
    print(f"  Coefficient: {top_pred['coefficient']:.4f}")
    print(f"  P-value: {top_pred['p_value']:.4f}")
else:
    top_pred = res_df.reindex(res_df['coefficient'].abs().sort_values(ascending=False).index).iloc[0]
    print(f"\nTop predictor (not sig): {top_pred['variable']}")
    print(f"  Coefficient: {top_pred['coefficient']:.4f}")
    print(f"  P-value: {top_pred['p_value']:.4f}")

# Also run for Claude
print("\n" + "="*60)
print("REGRESSION ANALYSIS: Claude QoL Bias")
print("="*60)
reg_df2 = df.dropna(subset=['claude_qol_en','log_gdp_pc','internet_pct','log_wiki','log_pop'])
reg_df2 = reg_df2[reg_df2['continent'].isin(valid_continents)].copy()
formula2 = "claude_qol_en ~ log_gdp_pc + internet_pct + log_wiki + log_pop + C(continent)"
model2 = smf.ols(formula2, data=reg_df2).fit(cov_type='HC3')
print(model2.summary())

with open(f"{BASE}/data/processed/regression_results.txt", 'a') as f:
    f.write("\nClaude-3.5-Haiku QoL Bias Regression (HC3 Robust SE)\n")
    f.write("="*60 + "\n")
    f.write(str(model2.summary()))

# ─── Spatial Autocorrelation (Moran's I) ────────────────────────────────────
print("\n" + "="*60)
print("SPATIAL AUTOCORRELATION: Moran's I")
print("="*60)

try:
    from libpysal.weights import KNN
    from esda.moran import Moran

    spatial_df = df.dropna(subset=['gpt_qol_en','latitude','longitude']).copy()
    coords = list(zip(spatial_df['longitude'], spatial_df['latitude']))
    
    # KNN weights (k=8)
    w = KNN.from_array(coords, k=8)
    w.transform = 'r'  # row-standardize
    
    moran = Moran(spatial_df['gpt_qol_en'].values, w)
    print(f"Moran's I (GPT QoL EN): {moran.I:.4f}")
    print(f"Expected I: {moran.EI:.4f}")
    print(f"P-value (sim): {moran.p_sim:.4f}")
    print(f"Z-score: {moran.z_sim:.4f}")
    
    moran_results = {
        'moran_I': moran.I,
        'expected_I': moran.EI,
        'p_value': moran.p_sim,
        'z_score': moran.z_sim,
        'n_cities': len(spatial_df)
    }
    
    # Also for Claude
    spatial_df2 = df.dropna(subset=['claude_qol_en','latitude','longitude']).copy()
    coords2 = list(zip(spatial_df2['longitude'], spatial_df2['latitude']))
    w2 = KNN.from_array(coords2, k=8)
    w2.transform = 'r'
    moran2 = Moran(spatial_df2['claude_qol_en'].values, w2)
    print(f"\nMoran's I (Claude QoL EN): {moran2.I:.4f}")
    print(f"P-value (sim): {moran2.p_sim:.4f}")

except Exception as e:
    print(f"Spatial analysis error: {e}")
    moran_results = {'moran_I': None, 'p_value': None, 'z_score': None, 'n_cities': len(df)}

import json
with open(f"{BASE}/data/processed/moran_results.json", 'w') as f:
    json.dump(moran_results, f, indent=2)

print("\nBias analysis complete!")
print(f"R² (GPT model): {model.rsquared:.4f}")
print(f"R² (Claude model): {model2.rsquared:.4f}")
