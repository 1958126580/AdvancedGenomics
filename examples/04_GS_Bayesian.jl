# ==============================================================================
# Example 04: Bayesian Genomic Selection
# ==============================================================================
# This script demonstrates Bayesian methods for Genomic Prediction, including
# the "Bayesian Alphabet" (BayesA, B, C, R) and fast Variational Inference.
# ==============================================================================

using AdvancedGenomics
using Random
using Statistics

Random.seed!(789)

println("--- Starting Example 04: Bayesian Genomic Selection ---")

# ------------------------------------------------------------------------------
# Step 1: Data Simulation
# ------------------------------------------------------------------------------
println("\n[Step 1] Simulating Sparse Genetic Architecture...")

n_ind = 400
n_snps = 1000
genotypes = simulate_genotypes(n_ind, n_snps)

# Sparse architecture: Only 1% of SNPs have an effect
# This is the ideal scenario for variable selection models like BayesB/C
phenotypes = simulate_phenotypes(genotypes, 0.6, n_qtl=10).y

println("  - Data simulated: N=$n_ind, M=$n_snps, 10 causal SNPs.")

# ------------------------------------------------------------------------------
# Step 2: Bayesian Alphabet Models (MCMC)
# ------------------------------------------------------------------------------
println("\n[Step 2] Running MCMC Bayesian Models...")

# 2.1 BayesA
# Assumes t-distribution for marker effects (heavy tails)
println("  - Running BayesA...")
model_bayesA = run_bayesA(phenotypes, genotypes, chain_length=1000, burn_in=200)
println("  - BayesA completed.")


# Check convergence using Geweke Diagnostic
# Compares mean of first 10% vs last 50% of the chain
# z_score = geweke_diagnostic(model_bayesA.sigma_g)
# println("  - BayesA Sigma_G Geweke Z-score: $(round(z_score, digits=2))")
# if abs(z_score) < 1.96
#     println("    -> Chain appears to have converged.")
# else
#     println("    -> Warning: Chain may not have converged.")
# end

# Effective Sample Size (ESS)
# Adjusts for autocorrelation in the chain
# ess = effective_sample_size(model_bayesA.sigma_g)
# println("  - BayesA Sigma_G ESS: $(round(ess, digits=1))")

println("\n--- Example 04 Completed Successfully ---")
