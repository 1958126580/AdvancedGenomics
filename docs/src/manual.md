# User Manual

## Genome-Wide Association Studies (GWAS)

AdvancedGenomics.jl provides efficient tools for GWAS using Linear Mixed Models (LMM).

### Basic Usage

```julia
using AdvancedGenomics

# Load data
genotypes = load_genotypes("data.vcf")
phenotypes = load_phenotypes("traits.csv")

# Run GWAS
results = run_gwas(genotypes, phenotypes)

# Visualize
manhattan_plot(results)
```

## Genomic Selection (GS)

Predict breeding values using various models.

### GBLUP Example

```julia
# Build Genomic Relationship Matrix (GRM)
K = build_grm(genotypes)

# Train model
model = train_gblup(phenotypes, K)

# Predict
ebv = predict(model, new_genotypes)
```

## Deep Learning

Leverage the power of neural networks for genomic prediction.

### Genomic Transformer

```julia
using AdvancedGenomics
using Lux

# Define model
model = GenomicTransformer(
    vocab_size = 4,
    embed_dim = 64,
    num_heads = 4,
    num_layers = 2
)

# Train
train!(model, genotypes, phenotypes)
```
