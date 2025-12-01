# AdvancedGenomics.jl User Guide

Welcome to the comprehensive user guide for **AdvancedGenomics.jl**. This document is designed to walk you through the package's capabilities using the 10 provided example scripts. Each section explains the scientific background, the code implementation, and the expected results.

---

## Table of Contents

1. [Standard GWAS Pipeline](#1-standard-gwas-pipeline)
2. [Advanced GWAS Methods](#2-advanced-gwas-methods)
3. [Genomic Selection (Kernels & ML)](#3-genomic-selection-kernels--ml)
4. [Bayesian Genomic Selection](#4-bayesian-genomic-selection)
5. [Deep Learning for Genomics](#5-deep-learning-for-genomics)
6. [Complex Statistical Models](#6-complex-statistical-models)
7. [Breeding Program Optimization](#7-breeding-program-optimization)
8. [Multi-Omics Integration](#8-multi-omics-integration)
9. [Modern Statistical Methods](#9-modern-statistical-methods)
10. [HPC & Population Genetics](#10-hpc--population-genetics)

---

## 1. Standard GWAS Pipeline

**Script:** [01_GWAS_Standard.jl](https://github.com/1958126580/AdvancedGenomics/blob/main/examples/01_GWAS_Standard.jl)

### Overview

This tutorial establishes the baseline workflow for any Genome-Wide Association Study (GWAS). It covers the essential steps: data simulation, quality control (QC), population structure correction (PCA), association testing (FarmCPU), and visualization.

### Key Steps

**1. Quality Control (QC)**
We filter SNPs based on Minor Allele Frequency (MAF) to remove rare variants that may cause spurious associations, and Hardy-Weinberg Equilibrium (HWE) to identify genotyping errors.

```julia
# Filter MAF < 0.05
genotypes_qc, _ = filter_maf(genotypes, 0.05)

# Filter HWE p < 1e-6
p_hwe = hwe_test(Matrix{Float64}(genotypes_qc))
genotypes_qc = genotypes_qc[:, p_hwe .>= 1e-6]
```

**2. Population Structure (PCA)**
Population stratification can lead to false positives. We use Principal Component Analysis (PCA) to capture genetic diversity and use the top PCs as covariates.

```julia
# Wrap in GenotypeMatrix for optimized operations
G_obj = GenotypeMatrix(genotypes_qc, ids, snps)
pca = run_pca(G_obj, k=3)
covariates = pca.projections
```

**3. FarmCPU Algorithm**
FarmCPU (Fixed and Random Model Circulating Probability Unification) is a powerful method that iteratively switches between a Fixed Effect Model (testing SNPs) and a Random Effect Model (selecting covariates) to boost power and control false positives.

```julia
results = run_farmcpu(phenotypes, G_obj, covariates)
```

**4. Visualization**
We generate interactive Manhattan and QQ plots using `PlotlyJS`.

```julia
manhattan_plot_interactive(results.p_values, chr, pos)
qq_plot_interactive(results.p_values)
```

---

## 2. Advanced GWAS Methods

**Script:** [02_GWAS_Advanced.jl](https://github.com/1958126580/AdvancedGenomics/blob/main/examples/02_GWAS_Advanced.jl)

### Overview

This tutorial expands on standard GWAS to handle binary traits, rare variants, and multi-study integration.

### Key Features

- **Logistic GWAS**: Uses logistic regression for Case-Control (0/1) phenotypes.
- **Burden Tests**: Aggregates rare variants (MAF < 0.01) within a gene or region into a "burden score" to test for collective effects, increasing statistical power for rare alleles.
- **Meta-Analysis**: Combines summary statistics (Beta, SE) from multiple independent studies using the Inverse Variance Weighted (IVW) method to improve discovery power.
- **Fine-Mapping**: Calculates Posterior Inclusion Probabilities (PIP) to identify the specific causal SNP within a significant genomic locus.

---

## 3. Genomic Selection (Kernels & ML)

**Script:** [03_GS_Kernels_ML.jl](https://github.com/1958126580/AdvancedGenomics/blob/main/examples/03_GS_Kernels_ML.jl)

### Overview

Genomic Selection (GS) aims to predict the genetic merit of individuals based on their DNA. This example compares classical Kernel methods with modern Machine Learning approaches.

### Methods Compared

1.  **Kernel Ridge Regression (RKHS)**:

    - **Linear Kernel**: Equivalent to GBLUP (Genomic Best Linear Unbiased Prediction). Captures additive effects.
    - **Polynomial Kernel**: Captures broad epistatic (interaction) effects.
    - **Gaussian (RBF) Kernel**: Captures complex, local non-linearities.

2.  **Machine Learning**:
    - **Random Forest**: An ensemble of decision trees. robust to noise and outliers.
    - **Gradient Boosting (GBM)**: sequentially builds trees to correct errors of previous trees. Often achieves state-of-the-art accuracy.

---

## 4. Bayesian Genomic Selection

**Script:** [04_GS_Bayesian.jl](https://github.com/1958126580/AdvancedGenomics/blob/main/examples/04_GS_Bayesian.jl)

### Overview

Bayesian methods allow for different prior assumptions about marker effects, often leading to higher accuracy than GBLUP when the trait is controlled by a few major genes (oligogenic architecture).

### Models Implemented

- **BayesA**: Assumes marker effects follow a t-distribution (allows for some large effects).
- **BayesB**: A variable selection model where many markers have zero effect (spike at zero) and others follow a t-distribution (slab).
- **BayesCPi**: Estimates the proportion of markers with non-zero effects ($\pi$) from the data.
- **Bayesian Lasso**: Uses a Double-Exponential (Laplace) prior, promoting sparsity.

---

## 5. Deep Learning for Genomics

**Script:** [05_Deep_Learning.jl](https://github.com/1958126580/AdvancedGenomics/blob/main/examples/05_Deep_Learning.jl)

### Overview

This tutorial demonstrates how to apply cutting-edge Deep Learning architectures to genomic data using `Lux.jl`.

### Architectures

**1. Genomic Transformer**
Adapted from NLP (e.g., BERT/GPT), Transformers use **Self-Attention** mechanisms to weigh the importance of different parts of the DNA sequence relative to each other. This is ideal for capturing long-range Linkage Disequilibrium (LD) patterns.

```julia
model = GenomicTransformer(vocab_size=3, embed_dim=32, num_heads=4, num_layers=2)
```

**2. Graph Neural Networks (GNN)**
GNNs model data as a graph. Here, genes are nodes, and biological pathways (protein-protein interactions) are edges. The model learns to aggregate information from neighboring genes in the pathway.

**3. Explainable AI (XAI)**
Deep Learning models are often "black boxes". We use **Saliency Maps** to compute the gradient of the output with respect to the input, highlighting which SNPs were most influential in the model's decision.

---

## 6. Complex Statistical Models

**Script:** [06_Complex_Models.jl](https://github.com/1958126580/AdvancedGenomics/blob/main/examples/06_Complex_Models.jl)

### Overview

Real-world data is often complex. This script handles multi-variate and longitudinal data.

### Models

- **Multi-Trait LMM**: Jointly analyzes two traits (e.g., Yield and Drought Tolerance) to estimate their **Genetic Correlation**. This is crucial for indirect selection (selecting for trait A to improve trait B).
- **Random Regression Models (RRM)**: Used for "Repeated Measures" or "Longitudinal Data" (e.g., weight recorded at multiple ages). It models the genetic effect as a function of time using Legendre Polynomials.

---

## 7. Breeding Program Optimization

**Script:** [07_Breeding.jl](https://github.com/1958126580/AdvancedGenomics/blob/main/examples/07_Breeding.jl)

### Overview

This example is designed for breeders. It focuses on making optimal selection decisions to maximize genetic gain while maintaining diversity.

### Tools

- **Selection Index**: Combines estimated breeding values (EBVs) for multiple traits into a single index value, weighted by their economic importance.
- **Optimal Contribution Selection (OCS)**: A quadratic optimization problem. It finds the optimal number of offspring each candidate should produce to maximize the Selection Index of the next generation while constraining the rate of inbreeding (relationship).

---

## 8. Multi-Omics Integration

**Script:** [08_Multi_Omics.jl](https://github.com/1958126580/AdvancedGenomics/blob/main/examples/08_Multi_Omics.jl)

### Overview

Genetics is just one layer. This script integrates Transcriptomics (Gene Expression) and Metabolomics to improve prediction accuracy and causal inference.

### Techniques

- **Omics Integration**: Concatenates genomic, transcriptomic, and metabolomic matrices to train a joint prediction model.
- **Mendelian Randomization (MR)**: A causal inference technique. It uses SNPs as "Instrumental Variables" to determine if an exposure (e.g., high expression of Gene X) _causes_ an outcome (e.g., Disease), avoiding confounding factors.

---

## 9. Modern Statistical Methods

**Script:** [09_Modern_Stats.jl](https://github.com/1958126580/AdvancedGenomics/blob/main/examples/09_Modern_Stats.jl)

### Overview

Showcases advanced statistical and optimization techniques.

- **Conformal Prediction**: Instead of a single point prediction, it outputs a prediction set (interval) that is guaranteed to contain the true value with probability $1-\alpha$ (e.g., 95%).
- **Heuristic Optimization**: Solves complex, non-differentiable problems (like feature selection) using nature-inspired algorithms:
  - **Genetic Algorithms (GA)**: Evolution-based search.
  - **Simulated Annealing (SA)**: Physics-based cooling process.
  - **Ant Colony Optimization (ACO)**: Pheromone-based path finding.

---

## 10. HPC & Population Genetics

**Script:** [10_HPC_PopGen.jl](https://github.com/1958126580/AdvancedGenomics/blob/main/examples/10_HPC_PopGen.jl)

### Overview

Demonstrates high-performance computing capabilities and population genetics analysis.

### Features

- **Haplotype Phasing**: Uses a Hidden Markov Model (HMM) to infer the two parental haplotypes from unphased genotype data.
- **IBD Detection**: Identifies Identity-by-Descent segments, which are long stretches of DNA shared between individuals due to a recent common ancestor.
- **GPU Acceleration**:
  - **`build_grm_gpu`**: Constructs the Genomic Relationship Matrix on the GPU.
  - **`run_gwas_gpu`**: Performs massive parallel association testing on the GPU.
  - _Note: Requires an NVIDIA GPU and `CUDA.jl`._

---

**AdvancedGenomics.jl** is designed to be flexible. We encourage you to modify these examples and adapt them to your specific research needs. Happy coding!
