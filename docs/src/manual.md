# User Manual

This manual provides a detailed guide to the features of **AdvancedGenomics.jl**.

## Data Input/Output

AdvancedGenomics.jl supports various data formats, including PLINK (bed/bim/fam), VCF (via external tools or custom readers), and CSV.

### Reading Genotypes

```julia
# Read from PLINK binary files
G = read_plink("data/genotypes") # expects .bed, .bim, .fam

# Read from CSV
G = read_genotypes("data/genotypes.csv")
```

### Reading Phenotypes

```julia
df = read_phenotypes("data/phenotypes.csv")
y = df.trait1
```

## Genome-Wide Association Studies (GWAS)

We support standard Linear Mixed Models (LMM) and advanced algorithms like FarmCPU.

### Standard LMM

The standard LMM accounts for population structure using a Genomic Relationship Matrix (GRM).

```julia
# 1. Build GRM
K = build_grm(G)

# 2. Run GWAS
# y: phenotype vector
# X: covariates (optional)
gwas_results = run_gwas(G, y, K)
```

### FarmCPU

Fixed and Random Model Circulating Probability Unification (FarmCPU) is a powerful method to control false positives and negatives.

```julia
farmcpu_results = run_farmcpu(G, y)
```

### Logistic GWAS (Case-Control)

For binary traits (0/1), use logistic regression.

```julia
logistic_results = run_logistic_gwas(G, y_binary)
```

## Genomic Selection (GS)

Predict breeding values or complex phenotypes using various models.

### GBLUP (Genomic Best Linear Unbiased Prediction)

```julia
# Train model
model = run_lmm(G_train, y_train, K_train)

# Predict
y_pred = predict(model, G_test)
```

### Bayesian Methods

We support BayesA, BayesB, BayesC, and Bayesian Lasso.

```julia
# BayesB
bayes_model = run_bayesB(G_train, y_train; iterations=1000, burnin=200)
```

### Machine Learning

Use Random Forests or Gradient Boosting for non-linear effects.

```julia
# Random Forest
rf_model = random_forest(G_train, y_train; n_trees=500)
y_pred_rf = predict_rf(rf_model, G_test)
```

## Deep Learning

Leverage the power of deep learning for genomic data.

### Genomic Transformer

A Transformer-based model adapted for SNP sequences.

```julia
# Initialize model
model = GenomicTransformer(n_snps=10000, d_model=128, n_heads=4)

# Train
train_transformer!(model, G_train, y_train; epochs=50)
```

### Explainable AI (XAI)

Visualize which SNPs are driving the predictions in your deep learning model.

```julia
# Generate saliency map
saliency = saliency_map(model, G_sample)
```

## Visualization

AdvancedGenomics.jl provides built-in plotting functions.

### Manhattan Plot

```julia
manhattan_plot(gwas_results; title="GWAS Results", threshold=5e-8)
```

### QQ Plot

```julia
qq_plot(gwas_results)
```

### Interactive Plots

For web-based interactive exploration:

```julia
manhattan_plot_interactive(gwas_results)
```
