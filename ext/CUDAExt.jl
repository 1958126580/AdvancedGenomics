module CUDAExt

using AdvancedGenomics
using CUDA
using LinearAlgebra
using Statistics
using Distributions

"""
    run_gwas_gpu(y, G)

GPU-accelerated GWAS.
"""
function AdvancedGenomics.run_gwas_gpu(y::Vector{Float64}, G::AdvancedGenomics.GenotypeMatrix)
    @info "Running GWAS on GPU..."
    
    # Move data to GPU
    y_gpu = CuArray(y)
    M_gpu = CuArray(G.data)
    
    n, m = size(M_gpu)
    
    # Simple regression: y = x*b + e
    # b = (x'x)^-1 x'y
    # x is SNP vector.
    # We can compute x'y and x'x for all SNPs in parallel.
    
    # Center y
    y_mean = mean(y_gpu)
    y_c = y_gpu .- y_mean
    
    # Center M (naive, assuming no missing)
    # M_mean = mean(M_gpu, dims=1)
    # M_c = M_gpu .- M_mean
    
    # Compute correlations efficiently
    # r = (x'y) / sqrt(x'x * y'y)
    
    # Dot products
    # X'Y: (m x n) * (n x 1) -> (m x 1)
    # M_gpu is n x m.
    
    # We need M_gpu' * y_c
    # But M is not centered.
    # cov(x, y) = E[xy] - E[x]E[y]
    # sum(xy) - sum(x)sum(y)/n
    
    sum_y = sum(y_gpu)
    sum_sq_y = sum(y_gpu.^2)
    
    # Compute sum(x) and sum(x^2) and sum(xy) for all SNPs
    # This can be done with matrix multiplication
    
    ones_n = CUDA.ones(n)
    
    sum_x = M_gpu' * ones_n
    sum_sq_x = (M_gpu.^2)' * ones_n
    sum_xy = M_gpu' * y_gpu
    
    # Numerator of beta
    # S_xy = sum_xy - sum_x * sum_y / n
    S_xy = sum_xy .- (sum_x .* sum_y ./ n)
    
    # Denominator (S_xx)
    # S_xx = sum_sq_x - sum_x^2 / n
    S_xx = sum_sq_x .- (sum_x.^2 ./ n)
    
    # Beta
    beta = S_xy ./ S_xx
    
    # Standard Error?
    # RSS = S_yy - beta * S_xy
    S_yy = sum_sq_y - sum_y^2 / n
    RSS = S_yy .- beta .* S_xy
    
    dof = n - 2
    sigma2 = RSS ./ dof
    se = sqrt.(sigma2 ./ S_xx)
    
    t_stat = beta ./ se
    
    # P-values (using CPU for distribution)
    t_cpu = Array(t_stat)
    p_values = 2.0 .* ccdf.(TDist(dof), abs.(t_cpu))
    
    return p_values
end

"""
    build_grm_gpu(G)

GPU-accelerated GRM construction.
"""
function AdvancedGenomics.build_grm_gpu(G::AdvancedGenomics.GenotypeMatrix)
    @info "Building GRM on GPU..."
    
    M = G.data
    n, m = size(M)
    
    # Move to GPU
    M_gpu = CuArray(M)
    
    # Handle missing (naive mean imputation on GPU?)
    # For now assume no missing or pre-imputed.
    # If missing, we can't easily use CuArray directly without imputation.
    
    # Calculate allele frequencies
    # p = mean(M, dims=1) / 2
    p = mean(M_gpu, dims=1) ./ 2.0f0
    
    # Center Z
    # Z = M - 2p
    Z = M_gpu .- (2.0f0 .* p)
    
    # Scale k
    # k = 2 * sum(p * (1-p))
    k = 2.0f0 * sum(p .* (1.0f0 .- p))
    
    # GRM = Z * Z' / k
    # CUBLAS gemm
    K_gpu = (Z * Z') ./ k
    
    return Matrix(K_gpu) # Return to CPU
end

end
