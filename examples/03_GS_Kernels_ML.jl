# ==============================================================================
# Example 03: Genomic Selection with Kernels & Machine Learning
# ==============================================================================
# This script demonstrates Genomic Prediction using various kernel methods
# (Linear, RBF, Polynomial) and Machine Learning algorithms (Random Forest, GBM).
# ==============================================================================

using AdvancedGenomics
using Random
using Statistics
using LinearAlgebra

Random.seed!(456)

println("--- Starting Example 03: Genomic Selection (Kernels & ML) ---")

# ------------------------------------------------------------------------------
# Step 1: Data Preparation
# ------------------------------------------------------------------------------
println("\n[Step 1] Preparing Data...")

n_train = 800
n_test = 200
n_snps = 2000

# Simulate training and testing data
genotypes_train = simulate_genotypes(n_train, n_snps)
genotypes_test = simulate_genotypes(n_test, n_snps)

# Simulate phenotype with non-linear effects (Epistasis)
# y = G + G*G + e
g_train = sum(genotypes_train[:, 1:10], dims=2)[:] # Additive
g_train += (genotypes_train[:, 1] .* genotypes_train[:, 2]) * 2.0 # Epistasis
y_train = g_train + randn(n_train)

g_test = sum(genotypes_test[:, 1:10], dims=2)[:]
g_test += (genotypes_test[:, 1] .* genotypes_test[:, 2]) * 2.0
y_test = g_test + randn(n_test)

println("  - Training Set: $n_train individuals")
println("  - Testing Set: $n_test individuals")

# ------------------------------------------------------------------------------
# Step 2: Kernel Methods (GBLUP & RKHS)
# ------------------------------------------------------------------------------
println("\n[Step 2] Testing Kernel Methods...")

# 2.1 Linear Kernel (GBLUP)
# Captures additive genetic variance
# Wrap matrix in GenotypeMatrix
ids_train = ["Ind_$i" for i in 1:n_train]
snps_ids = ["SNP_$i" for i in 1:n_snps]
G_train = GenotypeMatrix(genotypes_train, ids_train, snps_ids)
# ==============================================================================
# Example 03: Genomic Selection with Kernels & Machine Learning
# ==============================================================================
# This script demonstrates Genomic Prediction using various kernel methods
# (Linear, RBF, Polynomial) and Machine Learning algorithms (Random Forest, GBM).
# ==============================================================================

using AdvancedGenomics
using Random
using Statistics
using LinearAlgebra

Random.seed!(456)

println("--- Starting Example 03: Genomic Selection (Kernels & ML) ---")

# ------------------------------------------------------------------------------
# Step 1: Data Preparation
# ------------------------------------------------------------------------------
println("\n[Step 1] Preparing Data...")

n_train = 800
n_test = 200
n_snps = 2000

# Simulate training and testing data
genotypes_train = simulate_genotypes(n_train, n_snps)
genotypes_test = simulate_genotypes(n_test, n_snps)

# Simulate phenotype with non-linear effects (Epistasis)
# y = G + G*G + e
g_train = sum(genotypes_train[:, 1:10], dims=2)[:] # Additive
g_train += (genotypes_train[:, 1] .* genotypes_train[:, 2]) * 2.0 # Epistasis
y_train = g_train + randn(n_train)

g_test = sum(genotypes_test[:, 1:10], dims=2)[:]
g_test += (genotypes_test[:, 1] .* genotypes_test[:, 2]) * 2.0
y_test = g_test + randn(n_test)

println("  - Training Set: $n_train individuals")
println("  - Testing Set: $n_test individuals")

# ------------------------------------------------------------------------------
# Step 2: Kernel Methods (GBLUP & RKHS)
# ------------------------------------------------------------------------------
println("\n[Step 2] Testing Kernel Methods...")

# 2.1 Linear Kernel (GBLUP)
# Captures additive genetic variance
# Wrap matrix in GenotypeMatrix
ids_train = ["Ind_$i" for i in 1:n_train]
snps_ids = ["SNP_$i" for i in 1:n_snps]
G_train = GenotypeMatrix(genotypes_train, ids_train, snps_ids)

K_linear = build_grm(G_train)
# run_lmm(y, X, K)
model_linear = run_lmm(y_train, ones(n_train, 1), K_linear)
# Predict on test set (simplified projection for demonstration)
# Note: In practice, we'd use the full K matrix including test individuals
y_pred_linear = mean(y_train) .+ zeros(n_test) # 2.3 Polynomial Kernel
# Captures higher-order interactions
K_poly = build_poly_kernel(genotypes_train, degree=2)
model_poly = run_lmm(y_train, ones(n_train, 1), K_poly)
println("  - Polynomial Kernel fitted.")

# ------------------------------------------------------------------------------
# Step 3: Machine Learning Methods
# ------------------------------------------------------------------------------
println("\n[Step 3] Testing Machine Learning Methods...")

# 3.1 Random Forest
# Non-linear, handles interactions automatically
# random_forest(X, y; n_trees, min_samples_split)
rf_model = random_forest(Matrix{Float64}(genotypes_train), Vector{Float64}(y_train), n_trees=100)
y_pred_rf = predict_rf(rf_model, genotypes_test)
acc_rf = cor(y_pred_rf, y_test)
println("  - Random Forest Accuracy: $(round(acc_rf, digits=4))")

# 3.2 Gradient Boosting
# Sequential ensemble of trees
println("  - Training Gradient Boosting Machine...")
gbm_model = gradient_boosting(genotypes_train, y_train, n_trees=50, lr=0.1)
y_pred_gbm = predict_gbm(gbm_model, genotypes_test)
acc_gbm = cor(y_pred_gbm, y_test)
println("    -> GBM Accuracy (r): $(round(acc_gbm, digits=4))")
println("  Note: ML models often outperform linear kernels when epistasis is present.")

println("\n--- Example 03 Completed Successfully ---")
