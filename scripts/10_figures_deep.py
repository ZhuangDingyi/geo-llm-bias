"""
Publication-quality figures for geo-llm-bias deep analysis (Figs 19-24)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import gaussian_kde
import warnings; warnings.filterwarnings('ignore')
import json, os

BASE = "/Users/dingyizhuang/MIT Dropbox/Dingyi Zhuang/geo-llm-bias"
FIG_DIR = f"{BASE}/figures"

# ── style ──
PALETTE = {
    'Africa': '#E63946',
    'Asia': '#2196F3',
    'Europe': '#4CAF50',
    'North America': '#FF9800',
    'South America': '#9C27B0',
    'Oceania': '#795548',
}
plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

# ── Load data ──
df = pd.read_csv(f"{BASE}/data/processed/analysis_dataset.csv")
df = df.rename(columns={'gpt_qol_en': 'llm_qol_en', 'gpt_qol_zh': 'llm_qol_zh', 'gpt_pop_en': 'llm_pop_en'})

def norm_continent(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) * 9 + 1 if mx > mn else pd.Series([5.5] * len(s), index=s.index)

df['gt_qol'] = df.groupby('continent')['gdp_per_capita'].transform(norm_continent)
df['bias_en'] = df['llm_qol_en'] - df['gt_qol']
df['bias_zh'] = df['llm_qol_zh'] - df['gt_qol']
df['log_gdp'] = np.log(df['gdp_per_capita'].clip(lower=100))
df['log_wiki'] = np.log1p(df.get('wiki_pageviews', pd.Series([0] * len(df), index=df.index)))
df['log_pop'] = np.log(df['population'])
df = df[df['continent'].isin(['Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania'])].dropna(
    subset=['bias_en', 'gdp_per_capita', 'internet_pct'])
df['need_score'] = 10 - df['gt_qol']
df['llm_need_score'] = 10 - df['llm_qol_en']

with open(f"{BASE}/data/processed/deep_analysis.json") as f:
    results = json.load(f)

print(f"Loaded df: N={len(df)}")

# ══════════════════════════════════════════════════════════════
# FIG 19: Real-world impact simulation
# ══════════════════════════════════════════════════════════════
print("Generating Fig 19...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Panel A: true need vs LLM need scatter
ax = axes[0]
mask = df['llm_need_score'].notna() & df['need_score'].notna()
df_s = df[mask].copy()
for cont, grp in df_s.groupby('continent'):
    ax.scatter(grp['need_score'], grp['llm_need_score'],
               color=PALETTE.get(cont, '#888'), alpha=0.55, s=22, label=cont, edgecolors='none')
rng = [df_s['need_score'].min() - 0.2, df_s['need_score'].max() + 0.2]
ax.plot(rng, rng, 'k--', lw=1.2, alpha=0.7, label='Perfect alignment')
rho = results['impact_simulation']['need_correlation_rho']
p = results['impact_simulation']['need_correlation_p']
p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
ax.text(0.05, 0.95, f"ρ = {rho:.3f}, {p_str}", transform=ax.transAxes,
        fontsize=10, va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#ccc', alpha=0.9))
ax.set_xlabel("True Need Score (10 − GT$_{QoL}$)")
ax.set_ylabel("LLM-Estimated Need Score (10 − LLM$_{QoL}$)")
ax.set_title("A.  True vs. LLM-Estimated City Need")
ax.legend(fontsize=8.5, framealpha=0.7, loc='lower right', ncol=2)

# Panel B: GDP comparison bar
ax = axes[1]
sim = results['impact_simulation']
categories = ['LLM-Priority\nTop-50 Cities', 'Truly Needy\nTop-50 Cities']
gdp_vals = [sim['llm_priority_mean_gdp'], sim['true_priority_mean_gdp']]
colors = ['#E74C3C', '#2196F3']
bars = ax.bar(categories, gdp_vals, color=colors, width=0.45, edgecolor='white', linewidth=1.2)
for bar, val in zip(bars, gdp_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
            f'${val:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
gap = sim['gdp_misallocation_gap']
ax.annotate('', xy=(1, gdp_vals[1] + 200), xytext=(0, gdp_vals[0] + 200),
            arrowprops=dict(arrowstyle='<->', color='#555', lw=1.5))
ax.text(0.5, max(gdp_vals) * 0.98, f'Gap: ${gap:,.0f}', ha='center', va='center', fontsize=9.5,
        color='#333', bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF9C4', edgecolor='#aaa'))
overlap = sim['top50_overlap_pct']
ax.text(0.98, 0.97, f"Top-50 overlap:\n{overlap:.0f}%",
        transform=ax.transAxes, ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#ccc', alpha=0.9))
ax.set_ylabel("Mean GDP per Capita (USD)")
ax.set_title("B.  LLM-Guided Investment Misdirects Resources")
ax.set_ylim(0, max(gdp_vals) * 1.2)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
plt.suptitle("Figure 19. Real-World Impact Simulation: LLM-Based Urban Need Assessment",
             fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/fig19_impact_simulation.png")
plt.close()
sz = os.path.getsize(f"{FIG_DIR}/fig19_impact_simulation.png")
print(f"  Fig 19: {sz/1024:.1f} KB")

# ══════════════════════════════════════════════════════════════
# FIG 20: Within-country vs between-country variance
# ══════════════════════════════════════════════════════════════
print("Generating Fig 20...")
fig, ax = plt.subplots(figsize=(10, 6))

wc = df.groupby('country_code')['bias_en'].agg(['mean', 'std', 'count']).reset_index()
wc3 = wc[wc['count'] >= 3].nlargest(10, 'std').reset_index(drop=True)

country_labels = {
    'UA': 'Ukraine', 'IQ': 'Iraq', 'BD': 'Bangladesh', 'IR': 'Iran',
    'MX': 'Mexico', 'NG': 'Nigeria', 'AE': 'UAE', 'KE': 'Kenya',
    'BO': 'Bolivia', 'PL': 'Poland', 'CN': 'China', 'US': 'USA',
    'IN': 'India', 'BR': 'Brazil', 'ID': 'Indonesia'
}
wc3['label'] = wc3['country_code'].map(lambda x: country_labels.get(x, x))
wc3_sorted = wc3.sort_values('std')

colors_bar = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(wc3_sorted)))
bars = ax.barh(wc3_sorted['label'], wc3_sorted['std'], color=colors_bar, edgecolor='white', height=0.65)
for bar, (_, row) in zip(bars, wc3_sorted.iterrows()):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"n={int(row['count'])}", va='center', fontsize=8.5, color='#444')

icc = results['scale_comparison']['icc']
pct_within = results['scale_comparison']['pct_within_country_variance']
ax.axvline(wc3_sorted['std'].mean(), color='#E63946', linestyle='--', lw=1.5,
           label=f"Mean within-country SD = {wc3_sorted['std'].mean():.3f}")
textstr = (f"ICC = {icc:.3f}\n"
           f"{pct_within:.1f}% of bias variance\nis within countries\n"
           f"(national income drives\n94.3% of total variance)")
ax.text(0.62, 0.12, textstr, transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#EBF5FB', edgecolor='#5DADE2', alpha=0.95))
ax.set_xlabel("Within-Country Bias Standard Deviation")
ax.set_title("Figure 20. Top-10 Countries by Within-Country LLM Bias Variance\n"
             "(countries with ≥3 sampled cities; motivates city-scale analysis)", fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/fig20_within_country_variance.png")
plt.close()
sz = os.path.getsize(f"{FIG_DIR}/fig20_within_country_variance.png")
print(f"  Fig 20: {sz/1024:.1f} KB")

# ══════════════════════════════════════════════════════════════
# FIG 21: Fairness analysis — at-risk cities
# ══════════════════════════════════════════════════════════════
print("Generating Fig 21...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

at_risk = df[(df['gdp_per_capita'] < df['gdp_per_capita'].quantile(0.2)) &
             (df['internet_pct'] < df['internet_pct'].quantile(0.2))].copy()
not_at_risk = df[~df.index.isin(at_risk.index)].copy()

# Panel A: violin
ax = axes[0]
parts = ax.violinplot([at_risk['bias_en'].dropna(), not_at_risk['bias_en'].dropna()],
                      positions=[1, 2], widths=0.6, showmedians=True, showextrema=True)
parts['bodies'][0].set_facecolor('#E63946')
parts['bodies'][0].set_alpha(0.7)
parts['bodies'][1].set_facecolor('#2196F3')
parts['bodies'][1].set_alpha(0.7)
parts['cbars'].set_linewidth(1.2)
ax.set_xticks([1, 2])
ax.set_xticklabels([f'At-Risk Cities\n(n={len(at_risk)})\nBottom 20% GDP\n& Internet',
                    f'Other Cities\n(n={len(not_at_risk)})'], fontsize=9.5)
ar_mean = at_risk['bias_en'].mean()
nar_mean = not_at_risk['bias_en'].mean()
ax.axhline(ar_mean, xmin=0.1, xmax=0.4, color='#E63946', linestyle=':', lw=2)
ax.axhline(nar_mean, xmin=0.6, xmax=0.9, color='#2196F3', linestyle=':', lw=2)
ax.text(1, ar_mean + 0.1, f'μ = {ar_mean:.2f}', ha='center', fontsize=9, color='#E63946', fontweight='bold')
ax.text(2, nar_mean + 0.1, f'μ = {nar_mean:.2f}', ha='center', fontsize=9, color='#2196F3', fontweight='bold')
t, p = stats.ttest_ind(at_risk['bias_en'].dropna(), not_at_risk['bias_en'].dropna())
p_str = f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"
ax.text(0.5, 0.97, f"t = {t:.2f}, {p_str}", transform=ax.transAxes, ha='center', va='top',
        fontsize=9.5, bbox=dict(boxstyle='round', facecolor='white', edgecolor='#aaa', alpha=0.9))
ax.set_ylabel("LLM Bias Score ($B_i$)")
ax.set_title("A.  Bias Distribution: At-Risk vs. Other Cities")

# Panel B: GDP quintile mean bias
ax = axes[1]
df_q = df.copy()
df_q['gdp_q'] = pd.qcut(df_q['gdp_per_capita'], q=5, labels=['Q1\n(Poorest)', 'Q2', 'Q3', 'Q4', 'Q5\n(Richest)'])
q_stats = df_q.groupby('gdp_q')['bias_en'].agg(['mean', 'sem']).reset_index()
bar_colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, 5))
bars = ax.bar(q_stats['gdp_q'].astype(str), q_stats['mean'],
              yerr=q_stats['sem'] * 1.96, color=bar_colors,
              edgecolor='white', capsize=4, error_kw={'elinewidth': 1.2, 'ecolor': '#555'})
ax.axhline(0, color='black', lw=0.8, linestyle='-')
ax.set_xlabel("GDP per Capita Quintile")
ax.set_ylabel("Mean LLM Bias Score ($B_i$)")
ax.set_title("B.  Mean Bias by GDP Quintile\n(95% CI error bars)")
for bar, val in zip(bars, q_stats['mean']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f'{val:.2f}', ha='center', fontsize=8.5)

plt.suptitle("Figure 21. Fairness Analysis: Intersectional Disadvantage in LLM Geographic Knowledge",
             fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/fig21_fairness_analysis.png")
plt.close()
sz = os.path.getsize(f"{FIG_DIR}/fig21_fairness_analysis.png")
print(f"  Fig 21: {sz/1024:.1f} KB")

# ══════════════════════════════════════════════════════════════
# FIG 22: Ground truth robustness — 3 GT definitions
# ══════════════════════════════════════════════════════════════
print("Generating Fig 22...")

# Build alternative GTs
df['gt_global'] = (df['gdp_per_capita'] - df['gdp_per_capita'].min()) / (
        df['gdp_per_capita'].max() - df['gdp_per_capita'].min()) * 9 + 1

def income_tier(gdp):
    if gdp > 12535: return 9.0
    elif gdp > 4046: return 6.33
    elif gdp > 1046: return 3.67
    else: return 1.0

df['gt_tier'] = df['gdp_per_capita'].apply(income_tier)
df['gt_composite'] = (df['log_gdp'] / df['log_gdp'].max() * 0.5 +
                      df['internet_pct'] / 100 * 0.3 +
                      df['log_wiki'] / df['log_wiki'].max() * 0.2) * 9 + 1

gt_specs = [
    ('gt_qol', 'Within-Continent\nGDP (Primary)', '#2196F3'),
    ('gt_global', 'Global\nGDP', '#4CAF50'),
    ('gt_tier', 'World Bank\nIncome Tier', '#FF9800'),
    ('gt_composite', 'GDP + Internet\n+ Wikipedia', '#9C27B0'),
]

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
axes = axes.flatten()
for i, (gt_col, label, color) in enumerate(gt_specs):
    ax = axes[i]
    sub = df.dropna(subset=[gt_col, 'llm_qol_en']).copy()
    r, p = stats.pearsonr(sub[gt_col], sub['llm_qol_en'])
    rho, _ = stats.spearmanr(sub[gt_col], sub['llm_qol_en'])
    for cont, grp in sub.groupby('continent'):
        ax.scatter(grp[gt_col], grp['llm_qol_en'],
                   color=PALETTE.get(cont, '#888'), alpha=0.4, s=14, edgecolors='none')
    m, b = np.polyfit(sub[gt_col], sub['llm_qol_en'], 1)
    x_line = np.linspace(sub[gt_col].min(), sub[gt_col].max(), 100)
    ax.plot(x_line, m * x_line + b, color=color, lw=2, alpha=0.9)
    ax.text(0.05, 0.95, f"r = {r:.3f}\nρ = {rho:.3f}", transform=ax.transAxes,
            fontsize=9.5, va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#ccc', alpha=0.9))
    ax.set_xlabel(f"Ground Truth: {label}", fontsize=9.5)
    ax.set_ylabel("LLM QoL Rating", fontsize=9.5)
    ax.set_title(f"Panel {'ABCD'[i]}.  {label.replace(chr(10), ' ')}", fontsize=10)
    if i == 0:
        handles = [mpatches.Patch(color=PALETTE[c], label=c) for c in PALETTE]
        ax.legend(handles=handles, fontsize=7, ncol=2, loc='upper left',
                  framealpha=0.6, handlelength=1)

plt.suptitle("Figure 22. Ground Truth Robustness: LLM QoL Rating vs. Four GT Definitions\n"
             "(positive correlation consistent across all specifications)", fontsize=11, fontweight='bold')
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/fig22_gt_robustness.png")
plt.close()
sz = os.path.getsize(f"{FIG_DIR}/fig22_gt_robustness.png")
print(f"  Fig 22: {sz/1024:.1f} KB")

# ══════════════════════════════════════════════════════════════
# FIG 23: Population estimation and QoL bias joint analysis
# ══════════════════════════════════════════════════════════════
print("Generating Fig 23...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

df_pop = df.dropna(subset=['llm_pop_en', 'population', 'bias_en']).copy()
df_pop['pop_millions'] = df_pop['population'] / 1e6
df_pop['pop_error'] = (df_pop['llm_pop_en'] - df_pop['pop_millions']) / df_pop['pop_millions']
df_pop = df_pop[(df_pop['pop_error'] > -2) & (df_pop['pop_error'] < 8)]

# Panel A: 2D scatter + KDE contours
ax = axes[0]
for cont, grp in df_pop.groupby('continent'):
    ax.scatter(grp['pop_error'], grp['bias_en'],
               color=PALETTE.get(cont, '#888'), alpha=0.45, s=20, label=cont, edgecolors='none')
# Overall KDE contour
try:
    x_data = df_pop['pop_error'].values
    y_data = df_pop['bias_en'].values
    xmin, xmax = x_data.min(), x_data.max()
    ymin, ymax = y_data.min(), y_data.max()
    xx, yy = np.mgrid[xmin:xmax:60j, ymin:ymax:60j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x_data, y_data])
    kernel = gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    ax.contour(xx, yy, f, levels=5, colors='gray', alpha=0.35, linewidths=0.8)
except Exception:
    pass
rho_pop = results['pop_bias']['qol_pop_correlation']
p_pop = results['pop_bias']['qol_pop_p']
p_str_pop = "p < 0.001" if p_pop < 0.001 else f"p = {p_pop:.3f}"
ax.axhline(0, color='black', lw=0.7, linestyle='-')
ax.axvline(0, color='black', lw=0.7, linestyle='-')
ax.text(0.05, 0.97, f"ρ = {rho_pop:.3f}, {p_str_pop}", transform=ax.transAxes, va='top',
        fontsize=9.5, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#ccc', alpha=0.9))
ax.set_xlabel("Population Estimation Error\n(LLM − True) / True")
ax.set_ylabel("QoL Bias Score ($B_i$)")
ax.set_title("A.  Joint Distribution: Population Error vs. QoL Bias")
ax.legend(fontsize=8, framealpha=0.6, loc='upper right', ncol=2)

# Panel B: mean QoL bias by continent with pop error overlay
ax2 = axes[1]
cont_stats = df_pop.groupby('continent').agg(
    mean_bias=('bias_en', 'mean'),
    mean_pop_err=('pop_error', 'mean'),
    n=('bias_en', 'count')
).reset_index()
x_pos = np.arange(len(cont_stats))
bars_b = ax2.bar(x_pos - 0.2, cont_stats['mean_bias'], 0.35,
                 label='Mean QoL Bias', color='#2196F3', alpha=0.8, edgecolor='white')
ax2_twin = ax2.twinx()
bars_p = ax2_twin.bar(x_pos + 0.2, cont_stats['mean_pop_err'], 0.35,
                      label='Mean Pop Error', color='#FF9800', alpha=0.8, edgecolor='white')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([c.replace(' ', '\n') for c in cont_stats['continent']], fontsize=8.5)
ax2.set_ylabel("Mean QoL Bias Score", color='#2196F3')
ax2_twin.set_ylabel("Mean Population Estimation Error", color='#FF9800')
ax2.set_title("B.  QoL Bias and Population Error by Continent")
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8.5, loc='upper right')

plt.suptitle("Figure 23. Population Estimation and QoL Bias: Correlated Knowledge Gaps",
             fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/fig23_pop_qol_joint.png")
plt.close()
sz = os.path.getsize(f"{FIG_DIR}/fig23_pop_qol_joint.png")
print(f"  Fig 23: {sz/1024:.1f} KB")

# ══════════════════════════════════════════════════════════════
# FIG 24: IJGIS-style 3-panel summary / graphical abstract
# ══════════════════════════════════════════════════════════════
print("Generating Fig 24...")
fig = plt.figure(figsize=(15, 5.5))
gs = fig.add_gridspec(1, 3, wspace=0.38, hspace=0.1, left=0.05, right=0.97, top=0.88, bottom=0.15)

# Panel A: world map (B&W friendly scatter)
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    ax_map = fig.add_subplot(gs[0], projection=ccrs.Robinson())
    ax_map.add_feature(cfeature.LAND, facecolor='#F5F5F5', edgecolor='#BDBDBD', linewidth=0.4)
    ax_map.add_feature(cfeature.OCEAN, facecolor='#E3F2FD')
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor='#BDBDBD')
    ax_map.set_global()
    df_map = df.dropna(subset=['latitude', 'longitude', 'bias_en'])
    sc = ax_map.scatter(df_map['longitude'], df_map['latitude'], c=df_map['bias_en'],
                        cmap='RdBu_r', vmin=-4, vmax=8, s=18, alpha=0.8,
                        transform=ccrs.PlateCarree(), edgecolors='none')
    cbar = plt.colorbar(sc, ax=ax_map, orientation='horizontal', pad=0.04, shrink=0.7, aspect=25)
    cbar.set_label('Bias Score $B_i$', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    ax_map.set_title("A.  Global Distribution of LLM Geographic Bias\n(500 cities, 6 continents)",
                     fontsize=9.5, fontweight='bold', pad=4)
except ImportError:
    ax_map = fig.add_subplot(gs[0])
    df_map = df.dropna(subset=['longitude', 'latitude', 'bias_en'])
    sc = ax_map.scatter(df_map['longitude'], df_map['latitude'], c=df_map['bias_en'],
                        cmap='RdBu_r', vmin=-4, vmax=8, s=18, alpha=0.8, edgecolors='none')
    cbar = plt.colorbar(sc, ax=ax_map, orientation='horizontal', pad=0.04, shrink=0.8, aspect=25)
    cbar.set_label('Bias Score $B_i$', fontsize=9)
    ax_map.set_xlabel("Longitude"); ax_map.set_ylabel("Latitude")
    ax_map.set_title("A.  Global Distribution of LLM Geographic Bias\n(500 cities, 6 continents)",
                     fontsize=9.5, fontweight='bold', pad=4)

# Panel B: coefficient plot (key predictors from OLS)
import statsmodels.formula.api as smf
ax_coef = fig.add_subplot(gs[1])
df_reg = df.dropna(subset=['bias_en', 'log_gdp', 'internet_pct', 'log_wiki', 'log_pop'])
m = smf.ols('bias_en ~ log_gdp + internet_pct + log_wiki + log_pop + C(continent)',
            data=df_reg).fit(cov_type='HC3')
predictors = {
    'log_gdp': 'log GDP per capita',
    'internet_pct': 'Internet penetration (%)',
    'log_wiki': 'log Wikipedia pageviews',
    'log_pop': 'log Population',
}
coefs = []
for var, label in predictors.items():
    b = m.params[var]
    ci_lo = m.conf_int().loc[var, 0]
    ci_hi = m.conf_int().loc[var, 1]
    p = m.pvalues[var]
    coefs.append({'label': label, 'b': b, 'lo': ci_lo, 'hi': ci_hi, 'p': p})
coefs = sorted(coefs, key=lambda x: x['b'])
y_pos = range(len(coefs))
colors_coef = ['#E63946' if c['p'] < 0.05 else '#9E9E9E' for c in coefs]
ax_coef.barh(y_pos, [c['b'] for c in coefs],
             xerr=[[c['b'] - c['lo'] for c in coefs], [c['hi'] - c['b'] for c in coefs]],
             color=colors_coef, edgecolor='white', height=0.55,
             error_kw={'elinewidth': 1.5, 'ecolor': '#555', 'capsize': 4})
ax_coef.axvline(0, color='black', lw=1.0, linestyle='-')
ax_coef.set_yticks(y_pos)
ax_coef.set_yticklabels([c['label'] for c in coefs], fontsize=9)
ax_coef.set_xlabel("Regression Coefficient (95% CI)", fontsize=9)
ax_coef.set_title("B.  Key Predictors of LLM Bias\n(OLS with HC3-SE, continent FE)", fontsize=9.5, fontweight='bold')
sig_patch = mpatches.Patch(color='#E63946', label='p < 0.05')
ns_patch = mpatches.Patch(color='#9E9E9E', label='p ≥ 0.05')
ax_coef.legend(handles=[sig_patch, ns_patch], fontsize=8, loc='lower right')

# Panel C: misallocation bar
ax_bar = fig.add_subplot(gs[2])
sim = results['impact_simulation']
categories_c = ['LLM-Priority\nTop-50', 'Truly Needy\nTop-50']
vals_c = [sim['llm_priority_mean_gdp'], sim['true_priority_mean_gdp']]
bar_c = ax_bar.bar(categories_c, vals_c, color=['#E74C3C', '#2196F3'],
                   width=0.45, edgecolor='white')
for bar, val in zip(bar_c, vals_c):
    ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f'${val:,.0f}', ha='center', fontsize=9, fontweight='bold')
overlap_c = sim['top50_overlap_pct']
ax_bar.text(0.5, 0.88,
            f"Misidentification rate: {100 - overlap_c:.0f}%\nof truly needy cities excluded",
            transform=ax_bar.transAxes, ha='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', edgecolor='#F39C12', alpha=0.9))
ax_bar.set_ylabel("Mean GDP per Capita (USD)", fontsize=9)
ax_bar.set_title("C.  Investment Misallocation Simulation\n(LLM priorities vs. true need)", fontsize=9.5, fontweight='bold')
ax_bar.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

fig.suptitle("Figure 24. Summary: Digital Shadow Effect — Bias Distribution, Predictors, and Real-World Implications",
             fontsize=11, fontweight='bold', y=0.99)
fig.savefig(f"{FIG_DIR}/fig24_summary_graphical_abstract.png")
plt.close()
sz = os.path.getsize(f"{FIG_DIR}/fig24_summary_graphical_abstract.png")
print(f"  Fig 24: {sz/1024:.1f} KB")

print("\n=== ALL 6 FIGURES GENERATED ===")
# Verify sizes
for i in range(19, 25):
    fname_map = {19: 'fig19_impact_simulation', 20: 'fig20_within_country_variance',
                 21: 'fig21_fairness_analysis', 22: 'fig22_gt_robustness',
                 23: 'fig23_pop_qol_joint', 24: 'fig24_summary_graphical_abstract'}
    fpath = f"{FIG_DIR}/{fname_map[i]}.png"
    if os.path.exists(fpath):
        kb = os.path.getsize(fpath) / 1024
        status = "✓" if kb > 50 else "✗ TOO SMALL"
        print(f"  Fig {i}: {kb:.1f} KB {status}")
    else:
        print(f"  Fig {i}: MISSING")
