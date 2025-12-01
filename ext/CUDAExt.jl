module CUDAExt

using AdvancedGenomics
using CUDA
using LinearAlgebra
using Statistics

"""
    build_grm_gpu(G::GenotypeMatrix)

Constructs the Genomic Relationship Matrix on GPU using CUDA.
Implements VanRaden (2008) method: K = ZZ'/k where k = 2Σp(1-p).

This is a high-performance implementation optimized for NVIDIA GPUs.
"""
function AdvancedGenomics.build_grm_gpu(G::AdvancedGenomics.GenotypeMatrix)
    if !CUDA.functional()
        error("CUDA is not functional. Please check your GPU and CUDA installation.")
    end
    
    M = G.data
    n, m = size(M)
    
    # Convert to Float32 for GPU efficiency
    M_gpu = CuArray{Float32}(M)
    
    # Handle missing values with mean imputation on GPU
    # Compute column means
    col_sums = vec(sum(M_gpu, dims=1))
    col_counts = vec(sum(.!isnan.(M_gpu), dims=1))
    col_means = col_sums ./ col_counts
    
    # Replace NaN with means
    for j in 1:m
        mask = isnan.(view(M_gpu, :, j))
        M_gpu[mask, j] .= col_means[j]
    end
    
    # Compute allele frequencies
    p = col_means ./ 2.0f0
    
    # Center the matrix: Z = M - 2p
    Z = M_gpu .- (2.0f0 .* p')
    
    # Compute denominator: k = 2 * sum(p(1-p))
    k = 2.0f0 * sum(p .* (1.0f0 .- p))
    
    if k < 1f-6
        k = 1.0f0  # Avoid division by zero
    end
    
    # Compute GRM: K = ZZ'/k using CUDA BLAS
    # This is highly optimized on GPU
    K_gpu = CUDA.CUBLAS.gemm('N', 'T', 1.0f0/k, Z, Z)
    
    # Copy back to CPU and convert to Float64
    K = Array{Float64}(K_gpu)
    
    # Ensure symmetry (numerical precision)
    K = (K + K') / 2.0
    
    return K
end

"""
    run_gwas_gpu(y::Vector{Float64}, G::GenotypeMatrix)

Runs single-SNP GWAS on GPU for massive parallelization.
Computes t-statistics for each SNP in parallel.
"""
function AdvancedGenomics.run_gwas_gpu(y::Vector{Float64}, G::AdvancedGenomics.GenotypeMatrix)
    if !CUDA.functional()
        error("CUDA is not functional. Please check your GPU and CUDA installation.")
    end
    
    M = G.data
    n, m = size(M)
    
    # Transfer to GPU
    M_gpu = CuArray{Float32}(M)
    y_gpu = CuArray{Float32}(y)
    
    # Handle missing genotypes with mean imputation
    col_sums = vec(sum(M_gpu, dims=1))
    col_counts = vec(sum(.!isnan.(M_gpu), dims=1))
    col_means = col_sums ./ col_counts
    
    for j in 1:m
        mask = isnan.(view(M_gpu, :, j))
        M_gpu[mask, j] .= col_means[j]
    end
    
    # Compute statistics for each SNP in parallel
    # For each SNP j: beta_j = (X_j'X_j)^-1 X_j'y
    # t_j = beta_j / se_j
    
    # Vectorized computation
    XtX = vec(sum(M_gpu .^ 2, dims=1))  # m-vector
    Xty = vec(M_gpu' * y_gpu)           # m-vector
    
    beta = Xty ./ XtX
    
    # Residuals for each SNP
    # RSS_j = ||y - X_j * beta_j||^2
    y_sq = sum(y_gpu .^ 2)
    
    # Compute RSS efficiently
    # RSS = y'y - 2*beta*X'y + beta^2*X'X
    # But beta = X'y/X'X, so beta*X'X = X'y
    # RSS = y'y - beta*X'y
    RSS = y_sq .- beta .* Xty
    
    # Degrees of freedom
    dof = Float32(n - 2)
    
    # Variance
    sigma2 = RSS ./ dof
    
    # Standard errors
    se = sqrt.(sigma2 ./ XtX)
    
    # T-statistics
    t_stats = beta ./ se
    
    # P-values (approximate using normal distribution for speed)
    # For large n, t-distribution ≈ normal
    p_values = 2.0f0 .* CUDA.erfc.(abs.(t_stats) ./ sqrt(2.0f0))
    
    # Transfer back to CPU
    p_values_cpu = Array{Float64}(p_values)
    beta_cpu = Array{Float64}(beta)
    
    return (p_values=p_values_cpu, beta=beta_cpu, snps=G.snps)
end

end # module
