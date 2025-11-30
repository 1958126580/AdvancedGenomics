# ==============================================================================
# Example 06: Complex Statistical Models
# ==============================================================================
# This script demonstrates complex statistical models for specialized data types,
# including Multi-trait analysis, Threshold models for categorical data, and
# Random Regression for longitudinal data.
# ==============================================================================

using AdvancedGenomics
using Random
using Statistics
using LinearAlgebra
using DataFrames

Random.seed!(202)

println("--- Starting Example 06: Complex Statistical Models ---")

# ------------------------------------------------------------------------------
# Step 1: Multi-Trait Linear Mixed Model
# ------------------------------------------------------------------------------
println("\n[Step 1] Multi-Trait LMM (Genetic Correlation)...")

n_ind = 300
n_snps = 1000
genotypes = simulate_genotypes(n_ind, n_snps)

# Wrap in GenotypeMatrix for build_grm
ids = ["Ind_$i" for i in 1:n_ind]
snps_ids = ["SNP_$i" for i in 1:n_snps]
G_obj = GenotypeMatrix(genotypes, ids, snps_ids)

K = build_grm(G_obj)

# Step 2: Threshold Model for Categorical Traits
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
println("\n[Step 3] Random Regression Model (Repeated Measures)...")

# Simulate longitudinal data (e.g., weight measured at 3 time points)
n_time_points = 3
time_covariate = repeat(1:n_time_points, inner=n_ind)
# Stack genotypes for repeated measures structure
genotypes_rep = repeat(genotypes, outer=(n_time_points, 1))
y_rep = randn(n_ind * n_time_points)

# Run Random Regression
# Models the trajectory of genetic effects over time using Legendre polynomials
# run_random_regression(df, id_col, time_col, y_col, order)
# Create DataFrame
ids_rep = repeat(1:n_ind, inner=n_time_points)
df_rrm = DataFrame(ID=ids_rep, Time=time_covariate, Y=y_rep)

model_rrm = run_random_regression(df_rrm, :ID, :Time, :Y, 2)

println("  - Random Regression Model completed.")
println("  - Genetic variance estimated for intercept, slope, and curvature.")

println("\n--- Example 06 Completed Successfully ---")
