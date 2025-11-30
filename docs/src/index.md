# AdvancedGenomics.jl

Welcome to the documentation for **AdvancedGenomics.jl**, a high-performance genomic analysis package written in Julia.

## Features

- **GWAS**: Genome-Wide Association Studies using Linear Mixed Models.
- **Genomic Selection**: GBLUP, BayesB, and Machine Learning methods.
- **Simulation**: Realistic simulation of genomic data.
- **High Performance**: GPU acceleration and multi-threading support.

## Installation

```julia
using Pkg
Pkg.add("AdvancedGenomics")
```

## Quick Start

```julia
using AdvancedGenomics

# Simulate data
G = simulate_genotypes(100, 1000)
y = randn(100)

# Run GWAS
res = run_gwas_pipeline(G, y)
```
