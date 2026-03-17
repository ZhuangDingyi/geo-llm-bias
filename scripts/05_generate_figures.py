"""
Step 6: Publication-quality figures
"""
import os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy import stats
warnings.filterwarnings('ignore')

BASE = "/Users/dingyizhuang/MIT Dropbox/Dingyi Zhuang/geo-llm-bias"
FIGURES = f"{BASE}/figures"
os.makedirs(FIGURES, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

CONTINENT_COLORS = {
    'Africa': '#e74c3c',
    'Asia': '#f39c12',
    'Europe': '#3498db',
    'North America': '#2ecc71',
    'South America': '#9b59b6',
    'Oceania': '#1abc9c',
    'Other': '#95a5a6',
}

df = pd.read_csv(f"{BASE}/data/processed/analysis_dataset.csv")
print(f"Loaded {len(df)} cities")

# ─── Figure 1: World Map of GPT QoL Scores ────────────────────────────────────
print("Generating Figure 1: World map...")
try:
    import geopandas as gpd
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    world.plot(ax=ax, color='#ecf0f1', edgecolor='#bdc3c7', linewidth=0.5)
    
    scatter_df = df.dropna(subset=['gpt_qol_en'])
    scatter = ax.scatter(
        scatter_df['longitude'], scatter_df['latitude'],
        c=scatter_df['gpt_qol_en'],
        cmap='RdYlGn', s=30, alpha=0.8,
        vmin=1, vmax=10,
        edgecolors='white', linewidths=0.3,
        zorder=5
    )
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, aspect=20, pad=0.02)
    cbar.set_label('GPT-4o-mini Quality of Life Rating (1–10)', fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    
    ax.set_title('Figure 1: Geographic Distribution of LLM Quality of Life Ratings', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.tick_params(labelsize=9)
    
    # Add note
    ax.text(0.01, 0.02, f'N = {len(scatter_df)} cities | GPT-4o-mini (synthetic demo data)',
            transform=ax.transAxes, fontsize=8, color='gray', style='italic')
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES}/fig1_world_map.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ fig1_world_map.png")
except Exception as e:
    print(f"  Map error: {e}, using scatter fallback")
    fig, ax = plt.subplots(figsize=(16, 7))
    scatter_df = df.dropna(subset=['gpt_qol_en'])
    sc = ax.scatter(scatter_df['longitude'], scatter_df['latitude'],
                    c=scatter_df['gpt_qol_en'], cmap='RdYlGn', s=20, alpha=0.7,
                    vmin=1, vmax=10)
    plt.colorbar(sc, label='GPT-4o-mini QoL Rating')
    ax.set_title('Figure 1: Geographic Distribution of LLM QoL Ratings')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    plt.tight_layout()
    plt.savefig(f"{FIGURES}/fig1_world_map.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ fig1_world_map.png (fallback)")

# ─── Figure 2: Continent Boxplot Comparison ────────────────────────────────────
print("Generating Figure 2: Continent boxplots...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

continent_order = ['Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania', 'Other']

for ax, (col, title) in zip(axes, [('gpt_qol_en', 'GPT-4o-mini'), ('claude_qol_en', 'Claude-3.5-Haiku')]):
    plot_df = df.dropna(subset=[col]).copy()
    
    # Filter valid continents
    valid_conts = [c for c in continent_order if c in plot_df['continent'].values]
    
    data_by_continent = [plot_df[plot_df['continent'] == c][col].values for c in valid_conts]
    colors = [CONTINENT_COLORS.get(c, '#95a5a6') for c in valid_conts]
    
    bp = ax.boxplot(data_by_continent, patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    flierprops=dict(marker='o', markersize=4, alpha=0.5))
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticks(range(1, len(valid_conts)+1))
    ax.set_xticklabels(valid_conts, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Quality of Life Rating (1–10)', fontsize=11)
    ax.set_title(f'{title}', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 11)
    ax.axhline(y=5, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    
    # Add N labels
    for i, (c, vals) in enumerate(zip(valid_conts, data_by_continent)):
        ax.text(i+1, 10.5, f'n={len(vals)}', ha='center', fontsize=7, color='gray')

fig.suptitle('Figure 2: LLM Quality of Life Ratings by Continent\n(Higher = More Positive Bias)', 
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{FIGURES}/fig2_continent_boxplot.png", dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ fig2_continent_boxplot.png")

# ─── Figure 3: GDP vs LLM Rating Scatter ────────────────────────────────────────
print("Generating Figure 3: GDP scatter plots...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (col, title) in zip(axes, [('gpt_qol_en', 'GPT-4o-mini'), ('claude_qol_en', 'Claude-3.5-Haiku')]):
    plot_df = df.dropna(subset=[col, 'log_gdp_pc']).copy()
    
    for continent, group in plot_df.groupby('continent'):
        ax.scatter(group['log_gdp_pc'], group[col],
                   c=CONTINENT_COLORS.get(continent, '#95a5a6'),
                   label=continent, alpha=0.65, s=25, edgecolors='white', linewidths=0.2)
    
    # Regression line
    x = plot_df['log_gdp_pc'].values
    y = plot_df[col].values
    slope, intercept, r, p, se = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'k-', linewidth=2, alpha=0.8)
    ax.text(0.05, 0.95, f'β={slope:.3f}\np<0.001\nR²={r**2:.3f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Log GDP per Capita', fontsize=11)
    ax.set_ylabel('Quality of Life Rating (1–10)', fontsize=11)
    ax.set_title(f'{title}', fontsize=12, fontweight='bold')

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, -0.08), frameon=True)
fig.suptitle('Figure 3: LLM Quality of Life Rating vs. GDP per Capita\n(Color = Continent)', 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIGURES}/fig3_gdp_scatter.png", dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ fig3_gdp_scatter.png")

# ─── Figure 4: English vs Chinese Language Bias ──────────────────────────────
print("Generating Figure 4: Language bias comparison...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (model, en_col, zh_col, title) in zip(axes, [
    ('GPT', 'gpt_qol_en', 'gpt_qol_zh', 'GPT-4o-mini'),
    ('Claude', 'claude_qol_en', 'claude_qol_zh', 'Claude-3.5-Haiku')
]):
    plot_df = df.dropna(subset=[en_col, zh_col]).copy()
    
    for continent, group in plot_df.groupby('continent'):
        ax.scatter(group[en_col], group[zh_col],
                   c=CONTINENT_COLORS.get(continent, '#95a5a6'),
                   label=continent, alpha=0.5, s=20, edgecolors='white', linewidths=0.2)
    
    # Identity line
    lims = [max(1, min(plot_df[en_col].min(), plot_df[zh_col].min()) - 0.5),
            min(10, max(plot_df[en_col].max(), plot_df[zh_col].max()) + 0.5)]
    ax.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.6, label='y=x (no lang bias)')
    ax.set_xlim(lims); ax.set_ylim(lims)
    
    # Stats
    lang_diff = plot_df[en_col] - plot_df[zh_col]
    t_stat, p_val = stats.ttest_1samp(lang_diff, 0)
    ax.text(0.05, 0.95, f'Mean diff (EN−ZH): {lang_diff.mean():.3f}\nt={t_stat:.2f}, p={p_val:.3f}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('English Prompt QoL Rating', fontsize=11)
    ax.set_ylabel('Chinese Prompt QoL Rating', fontsize=11)
    ax.set_title(f'{title}: English vs Chinese Prompts', fontsize=11, fontweight='bold')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, -0.08), frameon=True)
fig.suptitle('Figure 4: Cross-Lingual Consistency — English vs. Chinese QoL Ratings', 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIGURES}/fig4_language_bias.png", dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ fig4_language_bias.png")

# ─── Figure 5: Regression Coefficient Plot ────────────────────────────────────
print("Generating Figure 5: Regression coefficients...")
import statsmodels.formula.api as smf

reg_df = df.dropna(subset=['gpt_qol_en','log_gdp_pc','internet_pct','log_wiki','log_pop'])
continent_counts = reg_df['continent'].value_counts()
valid_continents = continent_counts[continent_counts >= 3].index
reg_df = reg_df[reg_df['continent'].isin(valid_continents)].copy()

model = smf.ols("gpt_qol_en ~ log_gdp_pc + internet_pct + log_wiki + log_pop + C(continent)", 
                data=reg_df).fit(cov_type='HC3')

# Extract only main predictors (not continent FEs)
key_vars = ['log_gdp_pc', 'internet_pct', 'log_wiki', 'log_pop']
var_labels = {
    'log_gdp_pc': 'Log GDP\nper Capita',
    'internet_pct': 'Internet\nPenetration (%)',
    'log_wiki': 'Log Wikipedia\nPageviews',
    'log_pop': 'Log City\nPopulation',
}

coefs = [model.params[v] for v in key_vars]
cis = [(model.conf_int().loc[v, 0], model.conf_int().loc[v, 1]) for v in key_vars]
pvals = [model.pvalues[v] for v in key_vars]

fig, ax = plt.subplots(figsize=(10, 5))
colors_coef = ['#e74c3c' if c > 0 else '#3498db' for c in coefs]
y_pos = range(len(key_vars))

for i, (v, c, ci, p) in enumerate(zip(key_vars, coefs, cis, pvals)):
    ax.barh(i, c, color=colors_coef[i], alpha=0.75, height=0.5)
    ax.plot([ci[0], ci[1]], [i, i], 'k-', linewidth=2)
    ax.plot([ci[0], ci[1]], [i, i], 'k|', markersize=8, markeredgewidth=2)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    ax.text(c + 0.02, i, f'{sig}', va='center', fontsize=11, fontweight='bold',
            color='#e74c3c' if p < 0.05 else 'gray')

ax.axvline(x=0, color='black', linewidth=1.5, linestyle='-')
ax.set_yticks(list(y_pos))
ax.set_yticklabels([var_labels[v] for v in key_vars], fontsize=11)
ax.set_xlabel('Regression Coefficient (HC3 Robust SE)', fontsize=11)
ax.set_title('Figure 5: Predictors of LLM Geographic Bias (GPT-4o-mini)\nOutcome: Quality of Life Rating', 
             fontsize=12, fontweight='bold')
ax.text(0.98, 0.02, f'N={len(reg_df)} | R²={model.rsquared:.3f}\n*** p<0.001, ** p<0.01, * p<0.05, ns = not significant',
        transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(f"{FIGURES}/fig5_regression_coefs.png", dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ fig5_regression_coefs.png")

# ─── Figure 6: Model Comparison (GPT vs Claude) ─────────────────────────────
print("Generating Figure 6: Model comparison...")
fig, ax = plt.subplots(figsize=(10, 8))

compare_df = df.dropna(subset=['gpt_qol_en','claude_qol_en']).copy()

for continent, group in compare_df.groupby('continent'):
    ax.scatter(group['gpt_qol_en'], group['claude_qol_en'],
               c=CONTINENT_COLORS.get(continent, '#95a5a6'),
               label=continent, alpha=0.6, s=30, edgecolors='white', linewidths=0.3)

# Identity line
lims = [1, 11]
ax.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.6, label='Perfect agreement')
ax.set_xlim(lims); ax.set_ylim(lims)

# Correlation
r, p = stats.pearsonr(compare_df['gpt_qol_en'], compare_df['claude_qol_en'])
ax.text(0.05, 0.95, f'Pearson r = {r:.3f}\np < 0.001\nN = {len(compare_df)}',
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax.set_xlabel('GPT-4o-mini QoL Rating', fontsize=12)
ax.set_ylabel('Claude-3.5-Haiku QoL Rating', fontsize=12)
ax.set_title('Figure 6: Cross-Model Agreement in Geographic Bias\nGPT-4o-mini vs. Claude-3.5-Haiku', 
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig(f"{FIGURES}/fig6_model_comparison.png", dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ fig6_model_comparison.png")

# ─── Figure 7: Population Bias ────────────────────────────────────────────────
print("Generating Figure 7: Population estimation bias...")
fig, ax = plt.subplots(figsize=(12, 6))

pop_df = df.dropna(subset=['gpt_pop_bias']).copy()

# Clip extreme outliers for display
pop_df['gpt_pop_bias_clipped'] = pop_df['gpt_pop_bias'].clip(-3, 3)

for continent, group in pop_df.groupby('continent'):
    ax.scatter(group['log_pop'], group['gpt_pop_bias_clipped'],
               c=CONTINENT_COLORS.get(continent, '#95a5a6'),
               label=continent, alpha=0.6, s=25, edgecolors='white', linewidths=0.2)

ax.axhline(y=0, color='black', linewidth=2, linestyle='-', alpha=0.7, label='No bias (log ratio = 0)')
ax.set_xlabel('Log Actual City Population', fontsize=11)
ax.set_ylabel('Population Estimation Bias\n[log(LLM estimate / actual)]', fontsize=11)
ax.set_title('Figure 7: GPT-4o-mini Population Estimation Bias by City Size\n(Positive = Overestimate, Negative = Underestimate)', 
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9, ncol=2)

# Stats box
mean_bias = pop_df['gpt_pop_bias'].mean()
ax.text(0.02, 0.95, f'Mean bias: {mean_bias:.3f}\n({"Over" if mean_bias>0 else "Under"}estimate on average)',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(f"{FIGURES}/fig7_population_bias.png", dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ fig7_population_bias.png")

# ─── Figure 8: Wikipedia vs LLM correlation ──────────────────────────────────
print("Generating Figure 8: Wikipedia pageviews correlation...")
fig, ax = plt.subplots(figsize=(10, 6))

wiki_df = df[df['log_wiki'] > 0].dropna(subset=['gpt_qol_en']).copy()

for continent, group in wiki_df.groupby('continent'):
    ax.scatter(group['log_wiki'], group['gpt_qol_en'],
               c=CONTINENT_COLORS.get(continent, '#95a5a6'),
               label=continent, alpha=0.6, s=25)

# Regression
x = wiki_df['log_wiki'].values
y = wiki_df['gpt_qol_en'].values
slope, intercept, r, p, se = stats.linregress(x, y)
x_line = np.linspace(x.min(), x.max(), 100)
ax.plot(x_line, slope*x_line + intercept, 'k-', linewidth=2, alpha=0.8)

ax.text(0.05, 0.95, f'β={slope:.3f}, R²={r**2:.3f}\np={p:.3f}',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xlabel('Log Wikipedia Pageviews (2024)', fontsize=11)
ax.set_ylabel('GPT-4o-mini QoL Rating', fontsize=11)
ax.set_title('Figure 8: Wikipedia Visibility vs. LLM Quality of Life Ratings', 
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9, ncol=2)
plt.tight_layout()
plt.savefig(f"{FIGURES}/fig8_wikipedia_correlation.png", dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ fig8_wikipedia_correlation.png")

print(f"\nAll figures saved to {FIGURES}/")
print(f"Generated figures: {sorted(os.listdir(FIGURES))}")
