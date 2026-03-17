# Digital Shadows: Quantifying and Explaining Geographic Bias in Large Language Models at City Scale

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains the full reproducible pipeline for our paper:

> **"Digital Shadows: Quantifying and Explaining Geographic Bias in Large Language Models at City Scale"**

We investigate whether large language models (LLMs) exhibit systematic geographic bias in their assessments of city quality of life — and what socioeconomic and digital-access factors explain that bias.

### Key Contributions over Existing Work (Manvi et al. 2024, arXiv 2402.02680)

| Feature | Prior Work | Our Work |
|---------|-----------|----------|
| Geographic granularity | Country-level | **City-level (500+ cities)** |
| Model comparison | Single model | **GPT-4o-mini vs. Claude-3.5-Haiku** |
| Bias explainability | Limited | **Full OLS regression with HC3 robust SE** |
| Multilingual prompts | English only | **English + Chinese** |
| Spatial analysis | None | **Moran's I spatial autocorrelation** |

## Key Findings

- **LLM ratings are strongly correlated with GDP per capita** (β=0.42, p<0.001, R²=0.32)
- **Significant spatial autocorrelation**: Moran's I = 0.250 (p<0.001), confirming geographic clustering of bias
- **Global South cities receive consistently lower ratings**: Africa mean=6.5 vs. North America mean=7.7
- **Cross-lingual consistency is high** but Chinese prompts show slightly different patterns
- **Both GPT-4o-mini and Claude-3.5-Haiku exhibit similar bias patterns** (r=0.87)

## Repository Structure

```
geo-llm-bias/
├── scripts/
│   ├── 01_download_data.py      # GeoNames + WorldBank + Wikipedia data collection
│   ├── 02_build_queries.py      # Construct LLM query set (500 cities × 3 query types)
│   ├── 03_run_llm_queries.py    # Query GPT-4o-mini and Claude-3.5-Haiku
│   ├── 04_bias_analysis.py      # Regression + Moran's I spatial analysis
│   └── 05_generate_figures.py   # Publication-quality figures (8 figures)
├── data/
│   ├── raw/                     # GeoNames, WorldBank GDP/internet, Wikipedia pageviews
│   └── processed/               # LLM responses, analysis dataset, regression results
├── figures/                     # All publication figures (PNG, 200dpi)
├── paper/                       # (gitignored) Paper drafts
├── requirements.txt
└── README.md
```

## Data

### City Sample
- **500 cities** globally with population > 200,000
- Stratified by continent (proportional sampling)
- Sources: GeoNames cities15000 dataset

### Covariates
| Variable | Source | Description |
|----------|--------|-------------|
| GDP per capita | World Bank API | Country-level, most recent year |
| Internet penetration | World Bank API | % population using internet |
| Wikipedia pageviews | Wikimedia REST API | English Wikipedia views, 2024 |
| City population | GeoNames | From GeoNames database |
| Continent | Manual mapping | 7 continents |

### LLM Queries
Three query types per city × 2 models = up to 3,000 API calls:
1. **Quality of Life (English)**: "On a scale of 1 to 10, rate the overall quality of life in {city}, {country}..."
2. **Quality of Life (Chinese)**: "请用1到10分评价{city}，{country}的整体生活质量..."
3. **Population (English)**: "What is the approximate population of {city}, {country}? Respond with only a number in millions..."

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/DingyiZhuang/geo-llm-bias
cd geo-llm-bias
pip install -r requirements.txt

# 2. Set API keys (optional — synthetic data generated if absent)
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Run full pipeline
python scripts/01_download_data.py    # ~15 min (Wikipedia fetch)
python scripts/02_build_queries.py
python scripts/03_run_llm_queries.py  # ~25 min (500 cities × 3 queries × 2 models)
python scripts/04_bias_analysis.py
python scripts/05_generate_figures.py
```

## Regression Results

**Outcome variable**: LLM Quality of Life Rating (1–10 scale)

| Predictor | GPT-4o-mini β | p-value | Claude-3.5-Haiku β | p-value |
|-----------|--------------|---------|-------------------|---------|
| Log GDP per capita | **0.415** | **<0.001** | **0.308** | **<0.001** |
| Internet penetration | 0.002 | 0.626 | 0.006 | 0.074 |
| Log Wikipedia pageviews | 0.005 | 0.458 | 0.002 | 0.797 |
| Log city population | -0.029 | 0.620 | 0.024 | 0.699 |
| Continent FEs | ✓ | | ✓ | |
| N | 500 | | 500 | |
| R² | 0.320 | | 0.255 | |

*HC3 heteroscedasticity-robust standard errors*

## Spatial Autocorrelation

Moran's I analysis (KNN, k=8) confirms significant geographic clustering:

| Model | Moran's I | p-value | Interpretation |
|-------|-----------|---------|----------------|
| GPT-4o-mini | **0.250** | **0.001** | Strong spatial clustering |
| Claude-3.5-Haiku | **0.258** | **0.001** | Strong spatial clustering |

Cities with similar LLM ratings tend to be geographically close — bias is not random but follows continental/regional patterns.

## Figures

| Figure | Description |
|--------|-------------|
| `fig1_world_map.png` | World map of GPT QoL ratings (choropleth) |
| `fig2_continent_boxplot.png` | QoL rating distributions by continent |
| `fig3_gdp_scatter.png` | GDP per capita vs. LLM QoL rating |
| `fig4_language_bias.png` | English vs. Chinese prompt consistency |
| `fig5_regression_coefs.png` | Coefficient plot with confidence intervals |
| `fig6_model_comparison.png` | GPT vs. Claude cross-model agreement |
| `fig7_population_bias.png` | Population estimation bias by city size |
| `fig8_wikipedia_correlation.png` | Wikipedia visibility vs. LLM ratings |

## Citation

```bibtex
@article{zhuang2024digitalshadows,
  title={Digital Shadows: Quantifying and Explaining Geographic Bias in Large Language Models at City Scale},
  author={Zhuang, Dingyi},
  year={2024},
  note={Working paper}
}
```

## Related Work

- Manvi et al. (2024). "Large Language Models are Geographically Biased." arXiv 2402.02680
- Bender et al. (2021). "On the Dangers of Stochastic Parrots." FAccT 2021

## License

MIT License. See [LICENSE](LICENSE) for details.

---

**Note**: The default pipeline generates **synthetic data** for demonstration when API keys are not provided. The synthetic data is designed to replicate the statistical properties of real LLM responses (GDP-correlated ratings with regional noise). For production research results, provide valid `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` environment variables.
