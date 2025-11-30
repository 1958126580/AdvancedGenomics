# ==============================================================================
# Example 09: Modern Statistics & Optimization
# ==============================================================================
# This script demonstrates modern statistical techniques including Compressed Sensing
# (Lasso/ElasticNet), Conformal Prediction for uncertainty quantification,
# Wavelet Denoising, and heuristic Optimization algorithms.
# ==============================================================================

using AdvancedGenomics
using Random
using Statistics

Random.seed!(505)

println("--- Starting Example 09: Modern Statistics & Optimization ---")

# ------------------------------------------------------------------------------
# Step 1: Compressed Sensing (Lasso/ElasticNet)
# ------------------------------------------------------------------------------
println("\n[Step 1] Compressed Sensing (Coordinate Descent)...")

n = 100
p = 500
X = randn(n, p)
# Sparse signal: only 5 non-zero effects
beta_true = zeros(p)
beta_true[1:5] .= [1.0, -1.0, 0.5, -0.5, 2.0]
y = X * beta_true + randn(n) * 0.1

# Lasso (L1 regularization)
beta_lasso = lasso_cd(y, X, lambda=0.1)
println("  - Lasso completed. Non-zero coefs: $(count(beta_lasso .!= 0))")

# Elastic Net (L1 + L2 regularization)
beta_enet = elastic_net_cd(y, X, lambda=0.1, alpha=0.5)
println("  - Elastic Net completed. Non-zero coefs: $(count(beta_enet .!= 0))")

# ------------------------------------------------------------------------------
# Step 2: Conformal Prediction
# ------------------------------------------------------------------------------
println("\n[Step 2] Conformal Prediction (Uncertainty Intervals)...")

# Conformal prediction provides valid coverage guarantees for any predictive model
# We generate prediction intervals for the Lasso model
# Note: conformal_prediction splits data internally and uses Ridge regression
intervals = conformal_prediction(X, y, X, alpha=0.1) # 90% confidence

println("  - Conformal Prediction completed.")
println("  - Mean Interval Width: $(mean(intervals.upper - intervals.lower))")

# ------------------------------------------------------------------------------
# Step 3: Wavelet Denoising
# ------------------------------------------------------------------------------
println("\n[Step 3] Wavelet Denoising...")

# Simulate a noisy genomic signal (e.g., coverage depth)
signal = sin.(range(0, 10, length=100)) + randn(100) * 0.5
denoised_signal = wavelet_denoising(signal)

println("  - Wavelet Denoising completed.")
println("  - Noise Variance Reduction: $(round(var(signal) / var(denoised_signal), digits=2))x")

# ------------------------------------------------------------------------------
# Step 4: Heuristic Optimization
# ------------------------------------------------------------------------------
println("\n[Step 4] Heuristic Optimization Algorithms...")

# 4.1 Simulated Annealing (Continuous Optimization)
# Minimize a simple quadratic function: f(x) = sum(x^2)
println("  - Simulated Annealing (Minimizing sum(x^2))...")
obj_func(x) = sum(x.^2)
x0 = [10.0, -5.0, 3.0]
res_sa = simulated_annealing(obj_func, x0, max_iter=1000)
println("    -> Minimum found: $(round(res_sa.minimum, digits=4)) at $(round.(res_sa.minimizer, digits=2))")

# 4.2 Genetic Algorithm (Feature Selection)
# Select features to maximize R^2 in Ridge Regression
println("  - Genetic Algorithm (Feature Selection)...")
# Use the sparse signal data from Step 1
mask_ga = genetic_algorithm_select(X, y, pop_size=20, generations=10)
println("    -> GA selected $(sum(mask_ga)) features.")

# 4.3 Ant Colony Optimization (Feature Selection)
println("  - Ant Colony Optimization (Feature Selection)...")
mask_aco = ant_colony_select(X, y, n_ants=5, iterations=5, n_features=5)
println("    -> ACO selected $(sum(mask_aco)) features.")

println("\n--- Example 09 Completed Successfully ---")
