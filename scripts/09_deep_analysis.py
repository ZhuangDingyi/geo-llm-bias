"""
Deep analysis for geo-llm-bias paper
Adapted from task spec: uses analysis_dataset.csv directly (already merged)
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
import warnings; warnings.filterwarnings('ignore')
import json, os

BASE = "/Users/dingyizhuang/MIT Dropbox/Dingyi Zhuang/geo-llm-bias"

# ---- Load data ----
df = pd.read_csv(f"{BASE}/data/processed/analysis_dataset.csv")

# Rename columns to match expected names
df = df.rename(columns={
    'gpt_qol_en': 'llm_qol_en',
    'gpt_qol_zh': 'llm_qol_zh',
    'gpt_pop_en': 'llm_pop_en'
})

def norm_continent(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) * 9 + 1 if mx > mn else pd.Series([5.5] * len(s), index=s.index)

df['gt_qol'] = df.groupby('continent')['gdp_per_capita'].transform(norm_continent)
df['bias_en'] = df['llm_qol_en'] - df['gt_qol']
df['bias_zh'] = df['llm_qol_zh'] - df['gt_qol']
df['lang_diff'] = df['bias_zh'] - df['bias_en']
df['log_gdp'] = np.log(df['gdp_per_capita'].clip(lower=100))
df['log_wiki'] = np.log1p(df.get('wiki_pageviews', pd.Series([0] * len(df), index=df.index)))
df['log_pop'] = np.log(df['population'])
df = df[df['continent'].isin(['Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania'])].dropna(
    subset=['bias_en', 'gdp_per_capita', 'internet_pct'])
print(f"N = {len(df)}, continents: {df['continent'].value_counts().to_dict()}")

results = {}

# ================================================================
# EXP A: SPATIAL ERROR MODEL (primary specification for IJGIS)
# ================================================================
print("\n=== A. SPATIAL ERROR MODEL ===")
try:
    import geopandas as gpd
    from spreg import ML_Error
    from libpysal.weights import KNN

    df_s = df.dropna(subset=['bias_en', 'log_gdp', 'internet_pct', 'log_wiki', 'log_pop',
                              'latitude', 'longitude']).copy()
    gdf2 = gpd.GeoDataFrame(df_s, geometry=gpd.points_from_xy(df_s['longitude'],
                                                                df_s['latitude'])).set_crs('EPSG:4326')
    w2 = KNN.from_dataframe(gdf2, k=5)
    w2.transform = 'r'

    cont_dummies = pd.get_dummies(df_s['continent'], drop_first=True)
    X = pd.concat([df_s[['log_gdp', 'internet_pct', 'log_wiki', 'log_pop']], cont_dummies], axis=1)
    # Ensure all columns are numeric
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_cols = list(X.columns)
    X = sm.add_constant(X)
    y = df_s['bias_en'].values

    sem = ML_Error(y, X.values, w2)
    print(f"  Lambda: {sem.lam:.4f}")
    print(f"  Pseudo-R2: {sem.pr2:.4f}")
    coef_names = ['const'] + X_cols
    for i, (name, coef, se) in enumerate(zip(coef_names, sem.betas.flatten(), np.sqrt(np.diag(sem.vm)))):
        z = coef / se
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        print(f"  {name}: β={coef:.4f}, SE={se:.4f}, z={z:.3f}, p={p:.4f}")

    results['spatial_error'] = {
        'lambda': float(sem.lam), 'pseudo_r2': float(sem.pr2),
        'gdp_coef': float(sem.betas[1]), 'internet_coef': float(sem.betas[2]),
        'n': len(df_s)
    }
except Exception as e:
    print(f"  SEM error: {e}")
    # Fallback OLS
    m_ols = smf.ols('bias_en ~ log_gdp + internet_pct + log_wiki + log_pop + C(continent)',
                    data=df.dropna(subset=['bias_en', 'log_gdp', 'internet_pct', 'log_wiki', 'log_pop'])).fit(
        cov_type='HC3')
    print(f"  OLS fallback: R²={m_ols.rsquared:.4f}, GDP β={m_ols.params['log_gdp']:.4f}")
    results['spatial_error'] = {
        'lambda': None, 'pseudo_r2': float(m_ols.rsquared),
        'gdp_coef': float(m_ols.params['log_gdp']), 'internet_coef': float(m_ols.params['internet_pct']),
        'n': len(df)
    }

# ================================================================
# EXP B: PREDICTIVE FAIRNESS — who gets hurt most?
# ================================================================
print("\n=== B. PREDICTIVE FAIRNESS ANALYSIS ===")

df['gdp_quintile'] = pd.qcut(df['gdp_per_capita'], q=5,
                              labels=['Q1\n(poorest)', 'Q2', 'Q3', 'Q4', 'Q5\n(richest)'])
df['internet_quintile'] = pd.qcut(df['internet_pct'], q=5,
                                   duplicates='drop')

quint_stats = df.groupby('gdp_quintile')['bias_en'].agg(['mean', 'std', 'count']).reset_index()
print("Bias by GDP quintile:")
print(quint_stats.to_string())

at_risk = df[(df['gdp_per_capita'] < df['gdp_per_capita'].quantile(0.2)) &
             (df['internet_pct'] < df['internet_pct'].quantile(0.2))]
not_at_risk = df[~df.index.isin(at_risk.index)]
print(f"\n'At-risk' cities (bottom 20% GDP + internet): n={len(at_risk)}, mean bias={at_risk['bias_en'].mean():.3f}")
print(f"Other cities: n={len(not_at_risk)}, mean bias={not_at_risk['bias_en'].mean():.3f}")
t_stat, p_val = stats.ttest_ind(at_risk['bias_en'].dropna(), not_at_risk['bias_en'].dropna())
print(f"t-test: t={t_stat:.3f}, p={p_val:.6f}")

results['fairness'] = {
    'at_risk_n': len(at_risk), 'at_risk_mean_bias': float(at_risk['bias_en'].mean()),
    'other_mean_bias': float(not_at_risk['bias_en'].mean()),
    't_stat': float(t_stat), 'p_val': float(p_val),
    'bias_gap': float(not_at_risk['bias_en'].mean() - at_risk['bias_en'].mean()),
    'quint_stats': quint_stats.to_dict('records')
}

# ================================================================
# EXP C: REAL-WORLD IMPACT SIMULATION
# ================================================================
print("\n=== C. REAL-WORLD IMPACT SIMULATION ===")

df['need_score'] = 10 - df['gt_qol']
df['llm_need_score'] = 10 - df['llm_qol_en']

mask = df['llm_need_score'].notna() & df['need_score'].notna()
rho, p_rho = stats.spearmanr(df.loc[mask, 'need_score'], df.loc[mask, 'llm_need_score'])
print(f"Spearman correlation (true need vs LLM need): rho={rho:.4f}, p={p_rho:.6f}")

top50_llm = df.nlargest(50, 'llm_need_score')['geonameid'].tolist()
top50_true = df.nlargest(50, 'need_score')['geonameid'].tolist()
overlap = len(set(top50_llm) & set(top50_true))
print(f"Top-50 overlap (LLM vs true priority): {overlap}/50 = {overlap / 50 * 100:.1f}%")

gdp_llm_top50 = df[df['geonameid'].isin(top50_llm)]['gdp_per_capita'].mean()
gdp_true_top50 = df[df['geonameid'].isin(top50_true)]['gdp_per_capita'].mean()
print(f"Mean GDP of LLM-priority cities: ${gdp_llm_top50:,.0f}")
print(f"Mean GDP of truly-needy cities: ${gdp_true_top50:,.0f}")
print(f"GDP gap: ${gdp_llm_top50 - gdp_true_top50:,.0f} (LLM over-prioritizes wealthier cities)")

results['impact_simulation'] = {
    'need_correlation_rho': float(rho), 'need_correlation_p': float(p_rho),
    'top50_overlap_pct': overlap / 50 * 100,
    'llm_priority_mean_gdp': float(gdp_llm_top50),
    'true_priority_mean_gdp': float(gdp_true_top50),
    'gdp_misallocation_gap': float(gdp_llm_top50 - gdp_true_top50)
}

# ================================================================
# EXP D: COUNTRY-VS-CITY SCALE COMPARISON
# ================================================================
print("\n=== D. COUNTRY vs CITY SCALE COMPARISON ===")

within_country = df.groupby('country_code')['bias_en'].agg(['mean', 'std', 'count']).reset_index()
within_country = within_country[within_country['count'] >= 3]
print(f"Countries with ≥3 cities: {len(within_country)}")
print(f"Mean within-country SD: {within_country['std'].mean():.3f}")
print(f"Mean between-country range: {within_country['mean'].max() - within_country['mean'].min():.3f}")

top_variance = within_country.nlargest(10, 'std')[['country_code', 'mean', 'std', 'count']]
print("\nCountries with highest within-country bias variance:")
print(top_variance.to_string())

grand_mean = df['bias_en'].mean()
ss_between = sum(within_country['count'] * (within_country['mean'] - grand_mean) ** 2)
ss_within = sum(df.groupby('country_code')['bias_en'].apply(lambda x: ((x - x.mean()) ** 2).sum()))
icc = ss_between / (ss_between + ss_within)
print(f"\nIntraclass Correlation Coefficient (ICC): {icc:.4f}")
print(f"% variance within countries: {(1 - icc) * 100:.1f}%")
print(f"% variance between countries: {icc * 100:.1f}%")

results['scale_comparison'] = {
    'n_countries_3plus': len(within_country),
    'mean_within_country_sd': float(within_country['std'].mean()),
    'icc': float(icc),
    'pct_within_country_variance': float((1 - icc) * 100),
    'top_variance_countries': top_variance.to_dict('records')
}

# ================================================================
# EXP E: HDI PROXY ROBUSTNESS
# ================================================================
print("\n=== E. DEVELOPMENT TIER ROBUSTNESS ===")

def income_tier(gdp):
    if gdp > 12535:
        return 4
    elif gdp > 4046:
        return 3
    elif gdp > 1046:
        return 2
    else:
        return 1

df['income_tier'] = df['gdp_per_capita'].apply(income_tier)
df['gt_tier'] = df['income_tier'].map({1: 1, 2: 3.67, 3: 6.33, 4: 9})
df['bias_tier'] = df['llm_qol_en'] - df['gt_tier']

tier_stats = df.groupby('income_tier')['bias_tier'].agg(['mean', 'std', 'count']).reset_index()
print(tier_stats)

m_tier = smf.ols('bias_tier ~ log_gdp + internet_pct + log_wiki + C(continent)',
                 data=df.dropna(subset=['bias_tier', 'log_gdp', 'internet_pct'])).fit(cov_type='HC3')
print(f"Tier GT model: R²={m_tier.rsquared:.3f}, GDP β={m_tier.params['log_gdp']:.4f}, p={m_tier.pvalues['log_gdp']:.4f}")

df['composite_gt'] = (df['log_gdp'] / df['log_gdp'].max() * 0.5 +
                      df['internet_pct'] / 100 * 0.3 +
                      df['log_wiki'] / df['log_wiki'].max() * 0.2) * 9 + 1
df['bias_composite'] = df['llm_qol_en'] - df['composite_gt']
m_comp = smf.ols('bias_composite ~ log_gdp + internet_pct + log_wiki + C(continent)',
                 data=df.dropna(subset=['bias_composite', 'log_gdp', 'internet_pct'])).fit(cov_type='HC3')
print(f"Composite GT model: R²={m_comp.rsquared:.3f}, GDP β={m_comp.params['log_gdp']:.4f}, p={m_comp.pvalues['log_gdp']:.4f}")

results['gt_robustness'] = {
    'tier_model_r2': float(m_tier.rsquared), 'tier_gdp_beta': float(m_tier.params['log_gdp']),
    'composite_model_r2': float(m_comp.rsquared), 'composite_gdp_beta': float(m_comp.params['log_gdp'])
}

# ================================================================
# EXP F: POPULATION ESTIMATION SYSTEMATIC BIAS
# ================================================================
print("\n=== F. POPULATION ESTIMATION BIAS ===")
df_pop = df.dropna(subset=['llm_pop_en', 'population']).copy()
df_pop['pop_millions'] = df_pop['population'] / 1e6
df_pop['pop_error'] = (df_pop['llm_pop_en'] - df_pop['pop_millions']) / df_pop['pop_millions']

rho_pop, p_pop = stats.spearmanr(df_pop['pop_error'].clip(-2, 5), df_pop['bias_en'])
print(f"Pop error vs QoL bias correlation: rho={rho_pop:.4f}, p={p_pop:.4f}")

df_pop['log_gdp_p'] = np.log(df_pop['gdp_per_capita'].clip(lower=100))
m_pop = smf.ols('pop_error ~ log_gdp_p + internet_pct + log_wiki + C(continent)',
                data=df_pop.dropna(subset=['log_gdp_p', 'internet_pct'])).fit(cov_type='HC3')
print(f"Pop error model: R²={m_pop.rsquared:.3f}")
print(f"  GDP: β={m_pop.params['log_gdp_p']:.4f}, p={m_pop.pvalues['log_gdp_p']:.4f}")

results['pop_bias'] = {
    'qol_pop_correlation': float(rho_pop), 'qol_pop_p': float(p_pop),
    'pop_model_r2': float(m_pop.rsquared)
}

# Save
with open(f"{BASE}/data/processed/deep_analysis.json", 'w') as f:
    json.dump(results, f, indent=2, default=str)
print("\n=== ALL DEEP EXPERIMENTS DONE ===")
print(json.dumps({k: v for k, v in results.items() if isinstance(v, dict)}, indent=2, default=str))
