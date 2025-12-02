using AdvancedGenomics
using Test
using Random
using Statistics
using LinearAlgebra

println("Running Performance Verification...")

# 1. Verify IBS Kernel Optimization
println("\n1. Verifying IBS Kernel Optimization...")
n = 100
m = 1000
G_data = rand(0.0:2.0, n, m)
G = GenotypeMatrix(G_data, ["Ind$i" for i in 1:n], ["SNP$j" for j in 1:m])

println("Testing build_ibs_kernel...")
t_ibs = @elapsed K = build_ibs_kernel(G)
println("Time: $(round(t_ibs, digits=4))s")
@test size(K) == (n, n)
@test isapprox(K, K') # Symmetry check

# 2. Verify Bayesian Models Optimization
println("\n2. Verifying Bayesian Models Optimization...")
X = randn(n, 50) # Small feature set for quick test
true_beta = zeros(50)
true_beta[1:5] .= 1.0
y = X * true_beta + randn(n) * 0.5

println("Testing run_bayesA...")
t_bayesA = @elapsed res_A = run_bayesA(y, X; chain_length=500, burn_in=100)
println("BayesA Time: $(round(t_bayesA, digits=4))s")
@test length(res_A) == 50

println("Testing run_bayesC...")
t_bayesC = @elapsed res_C = run_bayesC(y, X; chain_length=500, burn_in=100)
println("BayesC Time: $(round(t_bayesC, digits=4))s")
@test length(res_C.beta) == 50

println("Testing run_bayesB...")
t_bayesB = @elapsed res_B = run_bayesB(y, X; chain_length=500, burn_in=100)
println("BayesB Time: $(round(t_bayesB, digits=4))s")
@test length(res_B.beta) == 50

println("Testing run_bayesR...")
t_bayesR = @elapsed res_R = run_bayesR(y, X; chain_length=500, burn_in=100)
println("BayesR Time: $(round(t_bayesR, digits=4))s")
@test length(res_R) == 50

println("\nAll performance tests passed!")
