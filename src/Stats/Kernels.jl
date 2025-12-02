"""
    Kernels.jl

Kernel functions for Genomic Prediction.
"""

using LinearAlgebra
using Statistics

"""
    build_dominance_kernel(G)

Constructs a Dominance Kernel Matrix.
"""
function build_dominance_kernel(G::GenotypeMatrix)
    M = G.data
    n, m = size(M)
    
    H = zeros(Float64, n, m)
    for j in 1:m
        col = view(M, :, j)
        # 1 is Het (assuming 0,1,2 coding)
        # We handle missing as 0 (no dominance effect)
        for i in 1:n
            if !isnan(col[i]) && col[i] == 1.0
                H[i, j] = 1.0
            end
        end
    end
    
    D = (H * H') ./ m
    return D
end

"""
    build_grm(G)

Constructs the Genomic Relationship Matrix (VanRaden 1).
"""
function build_grm(G::GenotypeMatrix)
    M = G.data
    n, m = size(M)
    
    # Handle missing with mean imputation
    M_imp = copy(M)
    for j in 1:m
        col = view(M, :, j)
        mu = mean(skipmissing(col))
        M_imp[isnan.(col), j] .= mu
    end
    
    # Center the matrix
    p = vec(mean(M_imp, dims=1)) ./ 2.0
    Z = (M_imp .- 2.0 .* p')
    k = 2.0 * sum(p .* (1.0 .- p))
    
    # Avoid div by zero
    if k == 0; k = 1.0; end
    
    # Efficient symmetric rank-k update: K = (1/k) * Z * Z'
    # BLAS.syrk!(uplo, trans, alpha, A, beta, C)
    # We compute the upper triangle and then symmetrize
    n = size(Z, 1)
    K = Matrix{Float64}(undef, n, n)
    BLAS.syrk!('U', 'N', 1.0/k, Z, 0.0, K)
    
    # Copy upper triangle to lower to make it fully symmetric
    LinearAlgebra.copytri!(K, 'U')
    return K
end



"""
    build_rbf_kernel(X; sigma=1.0)

Constructs a Radial Basis Function (Gaussian) Kernel.
"""
function build_rbf_kernel(X::Matrix{Float64}; sigma::Float64=1.0)
    n = size(X, 1)
    K = zeros(n, n)
    gamma = 1.0 / (2.0 * sigma^2)
    
    # Compute distance matrix efficiently
    # |x-y|^2 = |x|^2 + |y|^2 - 2x'y
    
    X2 = sum(X.^2, dims=2)
    D2 = X2 .+ X2' .- 2.0 .* (X * X')
    
    @. K = exp(-gamma * D2)
    return K
end

"""
    build_poly_kernel(X; degree=2, c=1.0)

Constructs a Polynomial Kernel: K(x,y) = (x'y + c)^d
"""
function build_poly_kernel(X::Matrix{Float64}; degree::Int=2, c::Float64=1.0)
    K = (X * X' .+ c).^degree
    return K ./ mean(diag(K))
end

"""
    build_interaction_kernel(K)

Constructs an Interaction Kernel (Hadamard product): K_int = K .* K
"""
function build_interaction_kernel(K::Matrix{Float64})
    return K .* K
end

"""
    build_ibs_kernel(G)

Constructs an Identity-By-State (IBS) Kernel.
Optimized using vectorized distance computation.
"""
function build_ibs_kernel(G::GenotypeMatrix)
    M = G.data
    n, m = size(M)
    
    # Handle missing values with mean imputation
    M_imp = copy(M)
    for j in 1:m
        col = view(M, :, j)
        mu = mean(skipmissing(col))
        M_imp[isnan.(col), j] .= mu
    end
    
    # Compute pairwise Manhattan distances efficiently
    # For genotype data (0,1,2), Manhattan distance = sum|x_i - x_j|
    # We can use: D_ij = sum_k |M_ik - M_jk|
    
    # Vectorized approach using broadcasting
    # D[i,j] = sum over k of |M[i,k] - M[j,k]|
    # This can be computed as: ||M[i,:] - M[j,:]||_1
    
    K = zeros(Float64, n, n)
    
    # Use BLAS for efficiency where possible
    # Compute all pairwise differences
    for i in 1:n
        # Vectorized computation for row i against all rows
        # Broadcasting: M_imp .- M_imp[i:i, :] gives (n x m) matrix
        diffs = abs.(M_imp .- M_imp[i:i, :])
        dists = vec(sum(diffs, dims=2))
        
        # Convert distance to similarity: IBS = 1 - dist/(2m)
        K[i, :] = 1.0 .- dists ./ (2.0 * m)
    end
    
    # Ensure perfect symmetry
    K = (K + K') / 2.0
    
    return K
end
