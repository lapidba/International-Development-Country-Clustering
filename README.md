# Multidimensional International Development Country Clustering (1964–2024)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.1%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Complete](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

> **A 60-Year Longitudinal Machine Learning Study of Peer Nations, Structural Bottlenecks, and Socioeconomic Transitions Across Six Decades Using the World Bank World Development Indicators (WDI).**

---

## Table of Contents
- [Executive Overview](#-executive-overview)
- [Research Question](#-research-question)
- [Repository Structure](#-repository-structure)
- [Dataset & Indicator Framework](#-dataset--indicator-framework)
- [Methodology & Machine Learning Pipeline](#-methodology--machine-learning-pipeline)
  - [Phase 1: Ingestion & Decade Windows](#phase-1-data-ingestion--decade-intervals)
  - [Phase 2: Data Cleaning & Preprocessing](#phase-2-data-cleaning--quality-engineering)
  - [Phase 3: Dimensionality Reduction (PCA)](#phase-3-dimensionality-reduction-pca)
  - [Phase 4: Unsupervised Clustering (K-Means)](#phase-4-unsupervised-clustering-k-means)
  - [Phase 5: Cross-Decade Transition Analysis](#phase-5-cross-decade-transition-analysis)
- [Key Empirical Findings](#-key-empirical-findings)
  - [Cluster Profiles (2014–2024 Modern Frontier)](#cluster-profiles-20142024-modern-frontier)
  - [60-Year Transition Dynamics](#60-year-transition-dynamics-19642024)
- [Policy Playbooks & Strategic Action Plans](#-policy-playbooks--strategic-action-plans)
- [Validation & Findings Criteria](#-validation--findings-criteria)
- [Installation & Quickstart](#-installation--quickstart)
- [Future Research Vectors](#-future-research-vectors)
- [License & Citation](#-license--citation)

---

## Executive Overview

For decades, international financial institutions (such as the World Bank and IMF) have categorized countries primarily through single-dimensional economic metrics—most notably **Gross National Income (GNI) per capita** (Low Income, Lower-Middle Income, Upper-Middle Income, and High Income).

While computationally simple, single-variable income thresholds obscure critical socioeconomic realities:
1. **The Middle-Income Trap & Inequality:** Countries with similar per-capita GDP often exhibit starkly divergent public health resilience, educational attainment, and infrastructure maturity.
2. **Resource-Rich Asymmetries:** Mineral- and petroleum-dependent economies often achieve high statistical GDP per capita while suffering from underdeveloped human capital and institutional infrastructure.
3. **Peer-Country Benchmarking:** Developing nations benefit significantly more from benchmarking policies against structural peers facing identical demographic or infrastructure constraints, rather than nations that merely share a GDP bracket.

This project delivers an **unsupervised machine learning framework** that analyzes **217 sovereign nations** across **18 canonical indicators** in **5 foundational pillars of development** over a **60-year horizon (1964–2024)** divided into six 10-year intervals.

---

## Research Question

> **Can we identify distinct, multidimensional developmental clusters of countries using economic, health, and social indicators — and how have those clusters shifted across ten-year intervals from 1964 to 2024 — to reveal "peer nations" that share similar developmental challenges despite varying income levels?**

---

## Repository Structure

```
.
├── Development_Clustering_Analysis.ipynb  # Master runnable Jupyter Notebook (end-to-end executed)
├── cluster_assignments_by_decade.csv      # Output dataset: Country cluster assignments (1964–2024)
├── requirements.txt                       # Python dependencies
├── Development-Clustering-Prompt-v2.md    # Project specification and analytical guidelines
├── data/                                  # Primary data directory (World Bank WDI)
│   ├── WDICSV.csv                         # Raw indicator time-series (1960–2025)
│   ├── WDICountry.csv                     # Country metadata (Region, Income Group)
│   ├── WDISeries.csv                      # Indicator definitions and topics
│   └── WDI_CSV.zip                        # Original archive from World Bank
├── plots/                                 # Exported publication-quality high-resolution figures
│   ├── 01_missingness_heatmap_grid.png
│   ├── 02_missingness_longitudinal_matrix.png
│   ├── 03_missingness_regional_disparities.png
│   ├── 04_eda_longitudinal_distributions.png
│   ├── 05_eda_correlation_matrices.png
│   ├── 06_pca_scree_plots.png
│   ├── 07_pca_component_loadings.png
│   ├── 08_kmeans_diagnostics_elbow_silhouette.png
│   ├── 09_pca_2d_clusters_1964_vs_2024.png
│   ├── 10_cross_decade_transition_heatmap.png
│   ├── 11_silhouette_longitudinal_cohesion.png
│   └── 12_cluster_centroids_standardized_profiles.png
└── README.md                              # Repository documentation
```
### Saved Visual Plot Catalog (`./plots/`)

| File Name | Description & Analytical Focus |
|---|---|
| [`01_missingness_heatmap_grid.png`](plots/01_missingness_heatmap_grid.png) | $2 \times 3$ grid of missingness matrices across all six 10-year intervals (1964–2024). |
| [`02_missingness_longitudinal_matrix.png`](plots/02_missingness_longitudinal_matrix.png) | Heatmap of 18 Indicators $\times$ 6 Decades displaying exact missingness percentages. |
| [`03_missingness_regional_disparities.png`](plots/03_missingness_regional_disparities.png) | Data availability disparity across 7 World Bank regions vs. the 40% threshold. |
| [`04_eda_longitudinal_distributions.png`](plots/04_eda_longitudinal_distributions.png) | Longitudinal boxplots for Life Expectancy, Infant Mortality, Urbanization, and GDP. |
| [`05_eda_correlation_matrices.png`](plots/05_eda_correlation_matrices.png) | Feature correlation heatmaps comparing baseline (1964–1973) vs modern (2014–2024). |
| [`06_pca_scree_plots.png`](plots/06_pca_scree_plots.png) | PCA scree plots & cumulative variance curves with $\ge 85\%$ threshold lines across all decades. |
| [`07_pca_component_loadings.png`](plots/07_pca_component_loadings.png) | Component loadings heatmap for PC1 (Modernization), PC2 (Growth), and PC3 (Social). |
| [`08_kmeans_diagnostics_elbow_silhouette.png`](plots/08_kmeans_diagnostics_elbow_silhouette.png) | Dual-axis Elbow (Inertia) & Silhouette score curves across $k \in [2, 10]$ per decade. |
| [`09_pca_2d_clusters_1964_vs_2024.png`](plots/09_pca_2d_clusters_1964_vs_2024.png) | 2D PCA cluster projections comparing structural clusters in 1964 vs 2014. |
| [`10_cross_decade_transition_heatmap.png`](plots/10_cross_decade_transition_heatmap.png) | 60-Year cross-decade country transition matrix and migration heatmap. |
| [`11_silhouette_longitudinal_cohesion.png`](plots/11_silhouette_longitudinal_cohesion.png) | Longitudinal clustering cohesion and silhouette score stability across six decades. |
| [`12_cluster_centroids_standardized_profiles.png`](plots/12_cluster_centroids_standardized_profiles.png) | Standardized Z-score centroid deviation profiles for all three developmental tiers. |

---

## Dataset & Indicator Framework

The primary data source is the **World Bank World Development Indicators (WDI)** archive. Sovereign entities (217 nations) are isolated by retaining rows with a **non-null `Region`** in `WDICountry.csv` (filtering out all regional and global aggregates).

### Canonical 18 Indicators Across 5 Development Pillars

| Development Pillar | Indicator Name | WDI Series Code | Policy Significance |
|---|---|---|---|
| **Economic Dynamism** | GDP per capita (constant 2015 US$) | `NY.GDP.PCAP.KD` | Baseline economic productive capacity |
| **Economic Dynamism** | GDP growth (annual %) | `NY.GDP.MKTP.KD.ZG` | Macroeconomic momentum and cyclical expansion |
| **Economic Dynamism** | Inflation, consumer prices (annual %) | `FP.CPI.TOTL.ZG` | Monetary stability and purchasing power protection |
| **Economic Dynamism** | Trade (% of GDP) | `NE.TRD.GNFS.ZS` | Openness and global value chain integration |
| **Public Health** | Life expectancy at birth, total (years) | `SP.DYN.LE00.IN` | Aggregate longevity and survival outcomes |
| **Public Health** | Infant mortality rate (per 1,000 live births) | `SP.DYN.IMRT.IN` | Primary healthcare delivery and maternal-child health |
| **Public Health** | Health expenditure (% of GDP) | `SH.XPD.CHEX.GD.ZS` | Domestic prioritization of public health systems |
| **Education & Capital**| Secondary school enrollment (% gross) | `SE.SEC.ENRR` | Intermediate skills and workforce readiness |
| **Education & Capital**| Primary school enrollment (% gross) | `SE.PRM.ENRR` | Foundational literacy pipeline |
| **Education & Capital**| Government education expenditure (% of GDP) | `SE.XPD.TOTL.GD.ZS` | Fiscal commitment to human capital formation |
| **Education & Capital**| Adult literacy rate (% ages 15+) | `SE.ADT.LITR.ZS` | Cumulative baseline human capital stock |
| **Demographics** | Population growth (annual %) | `SP.POP.GROW` | Demographic pressure and expansion rate |
| **Demographics** | Urban population (% of total) | `SP.URB.TOTL.IN.ZS` | Structural transformation from agrarian to urban |
| **Demographics** | Age dependency ratio (% working-age) | `SP.POP.DPND` | Youth and elderly economic dependency burden |
| **Infrastructure** | Access to electricity (% of population) | `EG.ELC.ACCS.ZS` | Foundational physical energy infrastructure |
| **Infrastructure** | Individuals using the Internet (%) | `IT.NET.USER.ZS` | Digital economy and frontier connectivity |
| **Infrastructure** | Mobile cellular subscriptions (per 100 people) | `IT.CEL.SETS.P2` | Mobile telecommunications penetration |
| **Infrastructure** | Fixed telephone subscriptions (per 100 people) | `IT.MLT.MAIN.P2` | Historical telecommunications backbone |

> **Note on Governance Indicators:** Worldwide Governance Indicators (WGI) measuring Rule of Law and Control of Corruption are published in a separate dataset starting only from 1996. To ensure consistent 60-year longitudinal integrity (1964–2024), governance metrics are excluded from the primary feature matrices and slated for future modular extension.

---

## Methodology & Machine Learning Pipeline

```
+----------------------------------------------------------------------------------------------------+
|                                    ANALYTICAL PIPELINE FLOW                                        |
+----------------------------------------------------------------------------------------------------+
  1. INGESTION        ──> Load WDI data & filter 217 sovereign countries
  2. DECADE MEANS     ──> Compute decade-mean matrices for six 10-year windows (1964–2024)
  3. CLEANING         ──> Deduplication | Missingness filtering (>40% feat, >50% country)
  4. PREPROCESSING    ──> 1st/99th percentile capping | Bounds [0, 100] | KNN Imputer (k=5)
  5. STANDARDIZATION  ──> StandardScaler (zero mean, unit variance)
  6. PCA              ──> Retain principal components explaining >= 85% cumulative variance
  7. K-MEANS          ──> Fit k=2..10 (random_state=42, init='k-means++', n_init=10)
  8. DIAGNOSTICS      ──> Elbow (Inertia) & Silhouette Coefficient analysis
  9. TRANSITIONS      ──> Construct cross-decade transition matrices & migration trajectories
 10. POLICY PLAYBOOK  ──> Tailored intervention frameworks for Governments, Multilaterals, & NGOs
+----------------------------------------------------------------------------------------------------+
```

### Phase 1: Data Ingestion & Decade Intervals
The 60-year timeline is segmented into six distinct non-overlapping intervals:
- **1964–1973** (Early Post-Colonial & Industrial Expansion)
- **1974–1983** (Oil Shocks & Stagflation Era)
- **1984–1993** (Structural Adjustment & East Asian Manufacturing Surge)
- **1994–2003** (Post-Cold War Globalization & Early Internet)
- **2004–2013** (Commodity Supercycle & Mobile Leapfrogging)
- **2014–2024** (Digital Economy, Demographic Aging, & Green Transition)

### Phase 2: Data Cleaning & Quality Engineering
1. **Duplicate Elimination:** Verified zero duplicate `(Country Code, Indicator Code)` records.
2. **Missingness Thresholding:** Excluded indicators with **>40% missingness** within each interval (e.g., Internet in 1964) and nations with **>50% missingness**.
3. **Robust Outlier Capping:** Applied 1st and 99th percentile winsorization to prevent hyperinflation or microstate trade leverage from skewing Euclidean distances.
4. **Consistency & Boundary Checks:** Enforced domain bounds $[0, 100]$ for percentages and $[0, \infty)$ for demographic/mortality rates.
5. **Imputation via `KNNImputer`:** Applied distance-weighted KNN ($k=5$) to borrow information from multi-attribute socioeconomic peers.
6. **Feature Scaling via `StandardScaler`:** Standardized all features to mean 0 and variance 1 prior to PCA and distance-based clustering.

### Phase 3: Dimensionality Reduction (PCA)
- **Variance Retention Criterion:** Evaluated eigenvalues and retained the minimum components explaining **$\ge 85\%$ cumulative variance** (4 components in 1964–1973; 7–8 components in modern decades).
- **Semantic Axes:** 
  - **PC1:** Structural Modernization, Health & Basic Infrastructure (35–45% variance).
  - **PC2:** Macroeconomic Growth Momentum & Trade Openness (12–18% variance).
  - **PC3:** Demographic Burden vs. Public Social Expenditure (8–12% variance).

### Phase 4: Unsupervised Clustering (K-Means)
- Strictly deterministic configuration: `random_state=42`, `init='k-means++'`, `n_init=10`.
- Evaluated $k=2$ through $10$ via **Elbow (Inertia/WCSS)** and **Silhouette Coefficient Analysis**.
- Selected **$k=3$** to establish an intuitive, policy-actionable tripartite development hierarchy:
  - **Tier 0:** Structural Bottleneck / Low-Income Vulnerability
  - **Tier 1:** Emerging / Middle-Income Transition
  - **Tier 2:** Advanced / High-Income Knowledge Economies

---

## Key Empirical Findings

### Cluster Profiles (2014–2024 Modern Frontier)

```
====================================================================================================
TIER 0: STRUCTURAL BOTTLENECK & HIGH VULNERABILITY (N = 43 Nations)
  - Life Expectancy: ~63.8 years | Infant Mortality: ~46.2 per 1,000 live births
  - Electricity Access: ~58.2% | Urban Population: ~37.4% | Internet Users: ~28.5%
  - Demographics: High Age Dependency (~78.4%) | High Annual Population Growth (~2.4%)
  - Structural Profile: High youth dependency burden, infrastructure deficits, maternal-child health constraints.
  - Exemplar Nations: Afghanistan, Chad, Central African Republic, Niger, Mali, Burundi, Somalia.

TIER 1: EMERGING & INTERMEDIATE TRANSITION (N = 104 Nations)
  - Life Expectancy: ~73.1 years | Infant Mortality: ~15.6 per 1,000 live births
  - Electricity Access: ~96.5% | Urban Population: ~61.8% | Internet Users: ~68.4%
  - Demographics: Moderate Age Dependency (~50.2%) | Low-to-Moderate Population Growth (~1.1%)
  - Structural Profile: Rapid urbanization, near-universal primary education, mobile-first digital adoption.
  - Exemplar Nations: Brazil, Colombia, Indonesia, Morocco, South Africa, Vietnam, Philippines, Egypt.

TIER 2: ADVANCED & HIGH-INCOME KNOWLEDGE ECONOMIES (N = 66 Nations)
  - Life Expectancy: ~80.9 years | Infant Mortality: ~3.8 per 1,000 live births
  - Electricity Access: 100.0% | Urban Population: ~81.2% | Internet Users: ~88.4%
  - Demographics: High Elderly Dependency (~53.8%) | Sub-Replacement Population Growth (~0.4%)
  - Structural Profile: Post-industrial knowledge and service economies, cutting-edge digital connectivity, aging demographics.
  - Exemplar Nations: Germany, Japan, Singapore, United Kingdom, United States, South Korea, Norway.
====================================================================================================
```

### 60-Year Transition Dynamics (1964–2024)

| 1964–1973 Baseline Tier | 2014–2024 Final Tier 0 (Low) | 2014–2024 Final Tier 1 (Emerging) | 2014–2024 Final Tier 2 (Advanced) | Total Sovereign Entities |
|---|:---:|:---:|:---:|:---:|
| **Tier 0 (Low Development)** | **36** *(Persistent Trap)* | **32** *(Upward Mobility)* | **0** | 68 |
| **Tier 1 (Emerging / Middle)** | **7** *(Conflict/Decay)* | **61** *(Stable Transition)* | **16** *(Advanced Ascent)* | 84 |
| **Tier 2 (Advanced Economies)**| **0** | **9** *(Macro De-leveraging)* | **45** *(Sustained Frontier)* | 54 |
| **Total Nations Clustered** | **43** | **102** | **61** | **206** |

#### Notable Trajectory Archetypes:
- **The Ascenders (Tier 1 $\rightarrow$ Tier 2):** South Korea, Singapore, Portugal, Chile, and Poland successfully transitioned into Tier 2 via massive secondary/tertiary human capital and high-tech infrastructure investments.
- **The Modernizers (Tier 0 $\rightarrow$ Tier 1):** Vietnam, Bangladesh, Bolivia, Indonesia, and Ghana broke out of the low-income trap through rural electrification, primary healthcare, and global supply chain integration.
- **The Persistent Bottleneck (Tier 0 $\rightarrow$ Tier 0):** Approximately 36 nations (predominantly in Sub-Saharan Africa and conflict-affected zones) remain constrained by high infant mortality, low electrification, and extreme age dependency.

---

## Policy Playbooks & Strategic Action Plans

### 1. For Sovereign Governments
- **Tier 0 (Structural Bottleneck):**
  - Allocate $\ge 15\%$ of national budgets to primary health (maternal-child health clinics, immunization) and clean water/sanitation to lower infant mortality.
  - Prioritize decentralized, off-grid renewable electrification for rural schools and clinics.
  - Enforce mandatory primary and lower-secondary education for girls to accelerate the demographic transition.
- **Tier 1 (Emerging Transition):**
  - Reform secondary and tertiary curricula to focus on STEM and vocational technical skills to escape the middle-income trap.
  - Invest in high-speed digital broadband and transport logistics to facilitate export diversification beyond raw commodities.
- **Tier 2 (Advanced Economies):**
  - Implement active labor market policies and healthcare productivity automation to mitigate the fiscal burden of demographic aging.
  - Accelerate smart grid modernization and net-zero green industrial transitions.

### 2. For Multilateral Lenders (World Bank, IMF, Regional Development Banks)
- **Multidimensional Concessionality:** Replace rigid single-variable GNI per capita thresholds with multidimensional cluster vulnerability scores for IDA concessional lending and climate resilience grants.
- **South-South Peer Matchmaking:** Facilitate bilateral peer-learning programs between ascending economies (e.g., Vietnam, Costa Rica) and structurally stalled Tier 0 peers.

### 3. For NGOs & Philanthropic Foundations
- Target high-leverage inflection points: secondary school retention grants for girls and solar mini-grids for agricultural processing.

---

## Validation & Findings Criteria

All empirical findings were subjected to explicit mathematical and domain validation thresholds:

| Validation Dimension | Explicit Threshold / Rule | Justification |
|---|---|---|
| **Silhouette Cohesion** | Average Silhouette Score $\ge 0.25$ | Guarantees genuine mathematical separation in feature space |
| **Cluster Size Viability** | Each cluster contains $\ge 5\%$ of sample ($\ge 10$ nations) | Prevents idiosyncratic outliers from masquerading as systemic clusters |
| **Longitudinal Stability** | Country trajectory consistency across $\ge 2$ consecutive decades | Distinguishes permanent structural shifts from short-term business-cycle shocks |
| **Domain Monotonicity** | Strictly monotonic ordering across Life Expectancy, Literacy, & Electricity | Confirms clusters align with real-world development hierarchies |

---

## Installation & Quickstart

### Prerequisites
- Python 3.10 or higher
- Git

### Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/lapidba/International-Development-Country-Clustering.git
cd International-Development-Country-Clustering

# 2. Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Launch Jupyter Notebook
jupyter notebook Development_Clustering_Analysis.ipynb
```

---

## Future Research Vectors

1. **Governance Integration (WGI):** Incorporate the World Bank Worldwide Governance Indicators (Rule of Law, Regulatory Quality, Control of Corruption) from 1996 onward.
2. **Climate Vulnerability Indices:** Integrate the Notre Dame Global Adaptation Initiative (ND-GAIN) index and carbon intensity per GDP unit.
3. **Non-Linear Manifold Learning:** Benchmark K-Means against non-linear algorithms (**UMAP + HDBSCAN** and Gaussian Mixture Models) to identify non-spherical micro-clusters.
4. **Sub-National Disaggregation:** Extend the clustering methodology to sub-national provincial and urban/rural datasets.

---


## Citation
```bibtex
@misc{international_development_clustering_2026,
  author = {Arjun Singh},
  title = {Multidimensional International Development Country Clustering (1964--2024): A 60-Year Longitudinal Machine Learning Study},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/lapidba/International-Development-Country-Clustering}}
}
```
