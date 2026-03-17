import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd, numpy as np, os, json
import statsmodels.formula.api as smf
import warnings; warnings.filterwarnings('ignore')

BASE = "/Users/dingyizhuang/MIT Dropbox/Dingyi Zhuang/geo-llm-bias"
FIG_DIR = f"{BASE}/figures"

plt.rcParams.update({
    'font.size': 12, 'font.family': 'DejaVu Sans',
    'axes.titlesize': 13, 'axes.labelsize': 12,
    'figure.dpi': 150, 'savefig.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10,
})

# ── Load processed results ──────────────────────────────────────────────────
ablation = pd.read_csv(f"{BASE}/data/processed/ablation_results.csv")
spatial  = pd.read_csv(f"{BASE}/data/processed/spatial_robustness.csv")
vif      = pd.read_csv(f"{BASE}/data/processed/vif_results.csv")

# Rebuild core dataset
df = pd.read_csv(f"{BASE}/data/processed/analysis_dataset.csv")
gpt = pd.read_csv(f"{BASE}/data/processed/llm_responses_gpt.csv")

qol_en = gpt[gpt['query_type']=='qol_en'][['geonameid','numeric_response']].rename(columns={'numeric_response':'llm_qol_en'})
qol_zh = gpt[gpt['query_type']=='qol_zh'][['geonameid','numeric_response']].rename(columns={'numeric_response':'llm_qol_zh'})
df = df.merge(qol_en, on='geonameid', how='left')
df = df.merge(qol_zh, on='geonameid', how='left')

def norm(s):
    mn, mx = s.min(), s.max()
    return (s-mn)/(mx-mn)*9+1 if mx > mn else pd.Series([5.5]*len(s), index=s.index)

df['gt_qol']       = df.groupby('continent')['gdp_per_capita'].transform(norm)
df['gt_qol_global']= norm(df['gdp_per_capita'])
df['bias_en']      = df['llm_qol_en'] - df['gt_qol']
df['bias_zh']      = df['llm_qol_zh'] - df['gt_qol']
df['bias_global']  = df['llm_qol_en'] - df['gt_qol_global']
df['lang_diff']    = df['bias_zh'] - df['bias_en']
df['bias_claude']  = df['claude_qol_en'] - df['gt_qol']
df = df[df['continent'] != 'Other'].dropna(subset=['bias_en','gdp_per_capita','internet_pct'])

CONT_COLORS = {
    'Africa':       '#d62728',
    'Asia':         '#ff7f0e',
    'Europe':       '#1f77b4',
    'North America':'#2ca02c',
    'South America':'#9467bd',
    'Oceania':      '#8c564b',
}

# ────────────────────────────────────────────────────────────────────────────
# FIG 11 — Ablation Study (R² and AIC side by side)
# ────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# rename column
ablation_plot = ablation.rename(columns={'R2': 'R2_val'}) if 'R2' in ablation.columns else ablation.rename(columns={'R²': 'R2_val'})
r2_col = 'R2_val'

colors_ab = ['#1f77b4' if i==0 else '#aec7e8' for i in range(len(ablation_plot))]
specs     = ablation_plot['Specification'].tolist()

ax = axes[0]
bars = ax.barh(range(len(ablation_plot)), ablation_plot[r2_col], color=colors_ab, height=0.65, alpha=0.88)
ax.set_yticks(range(len(ablation_plot)))
ax.set_yticklabels(specs, fontsize=10)
ax.set_xlabel('R² (OLS with HC3-robust SE)', fontsize=12)
ax.set_title('(a) Model Fit by Specification', fontsize=13, fontweight='bold')
full_r2 = ablation_plot[r2_col].iloc[0]
ax.axvline(full_r2, color='#1f77b4', linestyle='--', alpha=0.6, lw=1.5, label=f'Full model R²={full_r2:.3f}')
for i, v in enumerate(ablation_plot[r2_col]):
    ax.text(v + 0.006, i, f'{v:.3f}', va='center', fontsize=9)
ax.legend(fontsize=10); ax.set_xlim(0, 0.80)

ax = axes[1]
ax.barh(range(len(ablation_plot)), ablation_plot['AIC'], color=colors_ab, height=0.65, alpha=0.88)
ax.set_yticks(range(len(ablation_plot)))
ax.set_yticklabels(specs, fontsize=10)
ax.set_xlabel('AIC (lower = better fit)', fontsize=12)
ax.set_title('(b) AIC by Specification', fontsize=13, fontweight='bold')
full_aic = ablation_plot['AIC'].iloc[0]
ax.axvline(full_aic, color='#1f77b4', linestyle='--', alpha=0.6, lw=1.5, label=f'Full model AIC={full_aic:.0f}')
for i, v in enumerate(ablation_plot['AIC']):
    ax.text(v + 5, i, f'{v:.0f}', va='center', fontsize=9)
ax.legend(fontsize=10)

plt.suptitle('Figure 11: Ablation Study — Impact of Each Model Component', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig11_ablation.png", dpi=150, bbox_inches='tight')
plt.close(); print("Fig 11 saved ✓")

# ────────────────────────────────────────────────────────────────────────────
# FIG 12 — GDP Nonlinearity (decile bins)
# ────────────────────────────────────────────────────────────────────────────
df['gdp_decile_num'] = pd.qcut(df['gdp_per_capita'], q=10, duplicates='drop', labels=False)
n_bins = df['gdp_decile_num'].nunique()
decile_stats = df.groupby('gdp_decile_num')['bias_en'].agg(['mean','sem','count']).reset_index()
gdp_meds     = df.groupby('gdp_decile_num')['gdp_per_capita'].median().reset_index()
decile_stats = decile_stats.merge(gdp_meds, on='gdp_decile_num')

fig, ax = plt.subplots(figsize=(11, 6))
x = decile_stats['gdp_decile_num'].values
y = decile_stats['mean'].values
se= decile_stats['sem'].values

ax.fill_between(x, y - 1.96*se, y + 1.96*se, alpha=0.20, color='#1f77b4', label='95% CI')
ax.plot(x, y, 'o-', color='#1f77b4', lw=2.5, ms=9, label='Mean bias (EN)')
ax.axhline(0, color='black', ls='--', lw=1.2, alpha=0.5, label='Zero bias')

xlabels = [f'D{int(r["gdp_decile_num"])+1}\n(${int(r["gdp_per_capita"]):,})' for _, r in decile_stats.iterrows()]
ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=9, rotation=15)
ax.set_xlabel('GDP per Capita Decile (Median Value)', fontsize=12)
ax.set_ylabel('Mean LLM Bias Score ± 95% CI', fontsize=12)
ax.set_title('Figure 12: Nonlinearity Test — Does Bias Scale Monotonically with GDP?\n'
             '(Testing log-linear assumption of main regression)', fontsize=13, fontweight='bold')
for _, row in decile_stats.iterrows():
    ax.text(row['gdp_decile_num'], row['mean'] + 0.10, f'n={int(row["count"])}',
            ha='center', fontsize=8, color='#555')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig12_gdp_nonlinearity.png", dpi=150, bbox_inches='tight')
plt.close(); print("Fig 12 saved ✓")

# ────────────────────────────────────────────────────────────────────────────
# FIG 13 — 2×2 Heterogeneity Grid
# ────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 13))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.36)

# (a) Bias by population quartile
ax1 = fig.add_subplot(gs[0, 0])
df['pop_q'] = pd.qcut(df['population'], q=4,
                      labels=['Small\n<500K','Med\n0.5–1M','Large\n1–2M','Mega\n>2M'])
pop_stats = df.groupby('pop_q')['bias_en'].agg(['mean','sem','count']).reset_index()
bar_cols = ['#d62728','#ff9896','#aec7e8','#1f77b4']
ax1.bar(range(4), pop_stats['mean'],
        yerr=1.96*pop_stats['sem'], color=bar_cols, alpha=0.85, capsize=7, width=0.6)
ax1.set_xticks(range(4)); ax1.set_xticklabels(pop_stats['pop_q'], fontsize=10)
ax1.axhline(0, color='black', ls='--', alpha=0.45)
ax1.set_ylabel('Mean Bias Score', fontsize=11)
ax1.set_title('(a) Bias by City Size', fontsize=12, fontweight='bold')
for i, row in pop_stats.iterrows():
    offset = 0.15 if row['mean'] >= 0 else -0.30
    ax1.text(i, row['mean'] + offset, f"n={int(row['count'])}", ha='center', fontsize=9)

# (b) GDP β within each size category
ax2 = fig.add_subplot(gs[0, 1])
size_labels, betas, pvals, ns = [], [], [], []
for q_label, grp in df.groupby('pop_q'):
    g2 = grp.dropna(subset=['bias_en','log_gdp_pc'])
    if len(g2) > 15:
        m = smf.ols('bias_en ~ log_gdp_pc', data=g2).fit(cov_type='HC3')
        size_labels.append(str(q_label).replace('\n', ' '))
        betas.append(float(m.params['log_gdp_pc']))
        pvals.append(float(m.pvalues['log_gdp_pc']))
        ns.append(len(g2))

sig_colors = ['#d62728' if p < 0.001 else '#ff9896' if p < 0.05 else '#cccccc' for p in pvals]
ax2.bar(range(len(betas)), betas, color=sig_colors, alpha=0.85, width=0.6)
ax2.set_xticks(range(len(betas))); ax2.set_xticklabels(size_labels, fontsize=10, rotation=10)
ax2.axhline(0, color='black', ls='--', alpha=0.45)
ax2.set_ylabel('log GDP β Coefficient', fontsize=11)
ax2.set_title('(b) GDP Effect Within Each City Size\n(DV: bias score; red = p<0.001)', fontsize=12, fontweight='bold')
for i, (b, p, n) in enumerate(zip(betas, pvals, ns)):
    star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    ax2.text(i, b + (0.03 if b >= 0 else -0.08), star, ha='center', fontsize=13, fontweight='bold')
    ax2.text(i, min(betas) - 0.12, f'n={n}', ha='center', fontsize=9, color='#555')

# (c) VIF diagnostics
ax3 = fig.add_subplot(gs[1, 0])
vif_labels = {'log_gdp_pc': 'log GDP/cap', 'internet_pct': 'Internet %',
              'log_wiki': 'log Wikipedia', 'log_pop': 'log Population'}
vif['label'] = vif['Variable'].map(vif_labels)
vif_colors = ['#d62728' if v > 10 else '#ff9896' if v > 5 else '#2ca02c' for v in vif['VIF']]
ax3.barh(range(len(vif)), vif['VIF'], color=vif_colors, alpha=0.85, height=0.6)
ax3.set_yticks(range(len(vif))); ax3.set_yticklabels(vif['label'], fontsize=11)
ax3.axvline(5,  color='#ff7f0e', ls='--', lw=1.5, label='VIF=5 (moderate)')
ax3.axvline(10, color='#d62728', ls='--', lw=1.5, label='VIF=10 (severe)')
for i, v in enumerate(vif['VIF']):
    ax3.text(v + 2, i, f'{v:.1f}', va='center', fontsize=10, fontweight='bold')
ax3.set_xlabel('Variance Inflation Factor (VIF)', fontsize=11)
ax3.set_title('(c) Multicollinearity Diagnostics\n(Log GDP and Internet % are severely collinear)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9, loc='lower right')
ax3.set_xlim(0, max(vif['VIF'])*1.18)

# (d) Moran's I robustness across k
ax4 = fig.add_subplot(gs[1, 1])
moran_col = "Moran_I" if "Moran_I" in spatial.columns else "Moran's I"
ax4.plot(spatial['k'], spatial[moran_col], 'o-', color='#9467bd', lw=2.5, ms=10, label="Moran's I")
ax4.fill_between(spatial['k'],
                 spatial[moran_col] * 0.92,
                 spatial[moran_col] * 1.08,
                 alpha=0.18, color='#9467bd', label='±8% band')
ax4.set_xlabel('Number of Nearest Neighbors (k)', fontsize=11)
ax4.set_ylabel("Moran's I", fontsize=11)
ax4.set_title("(d) Spatial Autocorrelation: Robustness\nto Weight Matrix Choice (all p=0.001)",
              fontsize=12, fontweight='bold')
p_col = 'p_value' if 'p_value' in spatial.columns else 'p-value'
for _, row in spatial.iterrows():
    ax4.text(row['k'], row[moran_col] + 0.015,
             f"I={row[moran_col]:.3f}", ha='center', fontsize=8.5)
ax4.set_ylim(0.3, 0.85); ax4.legend(fontsize=9)

plt.suptitle('Figure 13: Heterogeneity, Multicollinearity Diagnostics, and Spatial Robustness',
             fontsize=13, fontweight='bold', y=1.01)
plt.savefig(f"{FIG_DIR}/fig13_heterogeneity_grid.png", dpi=150, bbox_inches='tight')
plt.close(); print("Fig 13 saved ✓")

# ────────────────────────────────────────────────────────────────────────────
# FIG 14 — Language Effect Analysis
# ────────────────────────────────────────────────────────────────────────────
df_lang = df.dropna(subset=['lang_diff','gdp_per_capita','internet_pct'])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
for cont, grp in df_lang.groupby('continent'):
    ax.scatter(np.log(grp['gdp_per_capita']), grp['lang_diff'],
               c=CONT_COLORS.get(cont, 'gray'), alpha=0.55, s=22, label=cont)
ax.axhline(0, color='black', ls='--', lw=1.2, alpha=0.5, label='Zero difference')
x_v = np.log(df_lang['gdp_per_capita']).values
y_v = df_lang['lang_diff'].values
m_f = np.isfinite(x_v) & np.isfinite(y_v)
coeffs = np.polynomial.polynomial.polyfit(x_v[m_f], y_v[m_f], 1)
xl = np.linspace(x_v[m_f].min(), x_v[m_f].max(), 200)
yl = coeffs[0] + coeffs[1]*xl
ax.plot(xl, yl, 'k--', lw=2, label=f'OLS slope={coeffs[1]:.3f}')
ax.set_xlabel('log(GDP per capita)', fontsize=12)
ax.set_ylabel('Language Effect: ZH bias − EN bias', fontsize=12)
ax.set_title('Language Effect vs Economic Development\n(Positive = ZH prompt gives higher rating)',
             fontsize=12, fontweight='bold')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=8, ncol=2, loc='upper left')

ax = axes[1]
cont_lang = df_lang.groupby('continent')['lang_diff'].agg(['mean','sem','count']).reset_index()
cont_order = cont_lang.sort_values('mean')
c_cols = [CONT_COLORS.get(c,'gray') for c in cont_order['continent']]
ax.barh(range(len(cont_order)), cont_order['mean'],
        xerr=1.96*cont_order['sem'], color=c_cols, alpha=0.85, height=0.65, capsize=5)
ax.axvline(0, color='black', ls='--', lw=1.2, alpha=0.5)
ax.set_yticks(range(len(cont_order)))
ax.set_yticklabels([f"{r['continent']} (n={int(r['count'])})"
                    for _, r in cont_order.iterrows()], fontsize=11)
ax.set_xlabel('Mean Language Effect ± 95% CI', fontsize=12)
ax.set_title('Language Effect by Continent\n(ZH vs EN prompt differential)',
             fontsize=12, fontweight='bold')

plt.suptitle('Figure 14: Multilingual Analysis — English vs Chinese Prompt Effects',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig14_language_analysis.png", dpi=150, bbox_inches='tight')
plt.close(); print("Fig 14 saved ✓")

# ────────────────────────────────────────────────────────────────────────────
# FIG 15 — Cross-Model Comparison: GPT-4o-mini vs Claude
# ────────────────────────────────────────────────────────────────────────────
df_cm = df.dropna(subset=['bias_en', 'bias_claude'])
corr = df_cm[['bias_en','bias_claude']].corr().iloc[0,1]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter GPT vs Claude
ax = axes[0]
for cont, grp in df_cm.groupby('continent'):
    ax.scatter(grp['bias_en'], grp['bias_claude'],
               c=CONT_COLORS.get(cont,'gray'), alpha=0.5, s=20, label=cont)
lim_min = min(df_cm['bias_en'].min(), df_cm['bias_claude'].min()) - 0.5
lim_max = max(df_cm['bias_en'].max(), df_cm['bias_claude'].max()) + 0.5
ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', lw=1.5, label='y=x (perfect agreement)')
ax.set_xlabel('GPT-4o-mini Bias Score', fontsize=12)
ax.set_ylabel('Claude-3.5-Haiku Bias Score', fontsize=12)
ax.set_title(f'(a) Model Agreement: GPT vs Claude\n(r={corr:.3f}, N={len(df_cm)})',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=8, ncol=2)

# Continent-level mean bias comparison
ax = axes[1]
gpt_cont   = df_cm.groupby('continent')['bias_en'].mean()
claude_cont= df_cm.groupby('continent')['bias_claude'].mean()
conts = list(gpt_cont.index)
x = np.arange(len(conts)); w = 0.38
ax.bar(x - w/2, [gpt_cont[c]   for c in conts], width=w, label='GPT-4o-mini',
       color='#1f77b4', alpha=0.85)
ax.bar(x + w/2, [claude_cont[c] for c in conts], width=w, label='Claude-3.5-Haiku',
       color='#ff7f0e', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(conts, rotation=15, fontsize=10)
ax.axhline(0, color='black', ls='--', alpha=0.45)
ax.set_ylabel('Mean Bias Score', fontsize=12)
ax.set_title('(b) Continental Bias Comparison\nGPT-4o-mini vs Claude-3.5-Haiku',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)

plt.suptitle('Figure 15: Cross-Model Validation — Geographic Bias is Model-Agnostic',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig15_cross_model.png", dpi=150, bbox_inches='tight')
plt.close(); print("Fig 15 saved ✓")

# ────────────────────────────────────────────────────────────────────────────
# Summary report
# ────────────────────────────────────────────────────────────────────────────
print("\n=== FIGURE INVENTORY ===")
for f in sorted(os.listdir(FIG_DIR)):
    if f.endswith('.png'):
        sz = os.path.getsize(f'{FIG_DIR}/{f}')
        flag = '✓' if sz > 30_000 else '✗ TOO SMALL'
        print(f"  {flag}  {f}: {sz//1024} KB")
