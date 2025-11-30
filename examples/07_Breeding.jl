# ==============================================================================
# Example 07: Breeding Program Optimization
# ==============================================================================
# This script demonstrates tools for animal and plant breeding, including
# pedigree analysis, selection index construction, and optimal contribution selection.
# ==============================================================================

using AdvancedGenomics
using Random
using DataFrames

Random.seed!(303)

println("--- Starting Example 07: Breeding Program Optimization ---")

# ------------------------------------------------------------------------------
# Step 1: Pedigree Analysis (A-Matrix)
# ------------------------------------------------------------------------------
println("\n[Step 1] Building Numerator Relationship Matrix (A)...")

using CSV

# Define a simple pedigree
# ID, Sire, Dam
pedigree_df = DataFrame(
    ID = ["1", "2", "3", "4", "5", "6"],
    Sire = ["0", "0", "1", "1", "3", "4"], # 0 = Unknown
    Dam =  ["0", "0", "2", "2", "2", "5"]
)

# Write to CSV
CSV.write("examples/pedigree_example.csv", pedigree_df)

# Read Pedigree from CSV
pedigree = read_pedigree("examples/pedigree_example.csv", separator=',', header=true)

# Build A-Matrix
A = build_A(pedigree)
A_inv = build_A_inverse(pedigree)

println("  - A-Matrix built (6x6).")
println("  - Relationship between Ind 5 and 6: $(round(A[5, 6], digits=4))")

# ------------------------------------------------------------------------------
# Step 2: Selection Index
# ------------------------------------------------------------------------------
println("\n[Step 2] Constructing Selection Index...")

# Assume we want to improve two traits: Yield and Disease Resistance
# Economic weights ($ value per unit improvement)
weights = [10.0, 5.0]

# Estimated Breeding Values (EBVs) for 6 individuals and 2 traits
ebvs = randn(6, 2)

# Phenotypic Covariance Matrix (P)
P = [1.0 0.2; 0.2 1.0]
# Genetic Covariance Matrix (G)
G_cov = 0.5 .* P

# Calculate Index Weights (b = P^-1 G w)
b = selection_index(P, G_cov, weights)

# Calculate Index Score = w1*EBV1 + w2*EBV2 (Simplified)
# Actually Index = b' * (Phenotypes - mean)
# For demo, we just use weighted sum of EBVs as "Index Score"
index_scores = ebvs * weights

println("  - Selection Index calculated.")
println("  - Top Individual ID: $(argmax(index_scores))")
println("\n[Step 3] Optimal Contribution Selection...")

# OCS balances genetic gain (Index Score) with genetic diversity (Inbreeding)
# Maximize: c'EBV - lambda * c'Ac
# Subject to: sum(c) = 1, c >= 0

# Penalty for inbreeding (lambda) -> target_inbreeding?
# The function uses target_inbreeding constraint, not lambda penalty directly?
# optimal_contribution_selection(ebv, A; target_inbreeding, n_offspring)
# We'll use a target inbreeding coefficient of 0.05
contributions = optimal_contribution_selection(Vector{Float64}(index_scores), Matrix{Float64}(A), target_inbreeding=0.05)

println("  - OCS Optimization completed.")
println("  - Optimal Contributions: $(round.(contributions.contributions, digits=3))")
println("  - Sum of Contributions: $(sum(contributions.contributions))")

# ------------------------------------------------------------------------------
# Step 4: Predicting Genetic Gain
# ------------------------------------------------------------------------------
println("\n[Step 4] Predicting Genetic Gain...")

# Predict response to selection
# R = i * r * sigma_g
selection_intensity = 1.755 # Select top 10%
accuracy = 0.8
sigma_g = 10.0

gain = predict_genetic_gain(selection_intensity, accuracy, sigma_g)

println("  - Predicted Genetic Gain: $(round(gain, digits=2)) units")

println("\n--- Example 07 Completed Successfully ---")
