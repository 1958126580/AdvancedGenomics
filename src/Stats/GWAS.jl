"""
    GWAS.jl

Advanced GWAS methods.
Includes FarmCPU and PCA.
Optimized for performance using Matrix Projection (Frisch-Waugh-Lovell).
"""

using LinearAlgebra
using Statistics
using Distributions

"""
    run_pca(G; k=3)

Computes top k Principal Components of the Genotype Matrix.
Used for population stratification correction.
"""
function run_pca(G::GenotypeMatrix; k::Int=3)
    M = G.data
    n, m = size(M)
    
    # Standardize
    # Handle missing data by mean imputation for PCA
    M_imp = copy(M)
    for j in 1:m
        col = view(M, :, j)
        mu = mean(skipmissing(col))
        M_imp[isnan.(col), j] .= mu
    end
    
    p = vec(mean(M_imp, dims=1)) ./ 2.0
    # Avoid division by zero for monomorphic
    denom = sqrt.(2.0 .* p .* (1.0 .- p))
    denom[denom .== 0] .= 1.0
    
    Z = (M_imp .- (2.0 .* p')) ./ denom'
    
    # Efficiently:
    # If m > n, compute Gram matrix K = Z Z' / m (n x n)
    # If n > m, compute Covariance matrix C = Z' Z / n (m x m) -> usually n << m in GWAS? 
    # Actually usually n < m.
    
    K = (Z * Z') ./ m
    F = eigen(Symmetric(K))
    
    # Eigenvalues are sorted ascending in Julia
    # Take last k
    indices = n:-1:(n-k+1)
    pcs = F.vectors[:, indices]
    
    return (projections=pcs, eigenvalues=F.values[indices])
end

"""
    run_farmcpu(y, G, X_cov; max_iter=10)

FarmCPU: Fixed and Random Model Circulating Probability Unification.
Optimized implementation using Frisch-Waugh-Lovell (FWL) theorem.
"""
function run_farmcpu(y::Vector{Float64}, G::GenotypeMatrix, X_cov::Matrix{Float64}; max_iter::Int=10)
    n, m = size(G.data)
    
    # Handle missing genotype data: Impute with mean for now
    # In production, we might want better imputation, but for speed:
    M = copy(G.data)
    for j in 1:m
        col = view(M, :, j)
        mu = mean(skipmissing(col))
        M[isnan.(col), j] .= mu
    end
    
    # Current pseudo-QTNs (indices)
    qtns = Int[]
    p_values = ones(m)
    
    # Pre-allocate arrays
    new_p_values = zeros(m)
    
    for iter in 1:max_iter
        # 1. Fixed Effect Model (FEM)
        # Model: y = X_fixed * b + SNP * a + e
        # X_fixed = [X_cov, QTNs]
        
        if isempty(qtns)
            X_fixed = X_cov
        else
            X_fixed = hcat(X_cov, M[:, qtns])
        end
        
        # Optimization: Project out X_fixed
        # P = I - X (X'X)^-1 X'
        # We need to compute y* = P y and M* = P M
        # Then simple regression: y* = M*_j * a + e
        # a = (M*_j' M*_j)^-1 M*_j' y*
        # But M*_j' M*_j is scalar (sum of squares).
        
        # QR Decomposition is more stable
        # X_fixed = Q R
        # Q' X_fixed = R
        # P = I - Q Q'
        # y* = y - Q (Q' y)
        # M*_j = M_j - Q (Q' M_j)
        
        F_fact = qr(X_fixed)
        Q = Matrix(F_fact.Q) # n x k
        
        # Project y
        Qty = Q' * y
        y_star = y - Q * Qty
        
        # Project M (all SNPs)
        # This is expensive: n x m matrix mult.
        # Q' M is k x m.
        # M* = M - Q * (Q' M)
        
        QtM = Q' * M
        
        # Compute statistics for each SNP
        # We can parallelize this
        
        # Variance of y_star (residual variance of null model)
        # Actually, we need the t-test statistic.
        # t = beta / se
        # beta = (x' x)^-1 x' y
        # Here x is M*_j.
        # beta = (M*_j' M*_j)^-1 M*_j' y_star
        # se = sqrt(sigma2 * (M*_j' M*_j)^-1)
        # sigma2 = RSS / (n - k - 1)
        # RSS = (y_star - M*_j * beta)' (y_star - M*_j * beta)
        
        dof = n - size(X_fixed, 2) - 1
        
        Threads.@threads for j in 1:m
            # Construct M*_j
            # M_star_j = M[:, j] - Q * QtM[:, j]
            # Optimization: We don't need full vector M_star_j if we just want dot products.
            # But we need it for RSS.
            
            # Let's compute M_star_j explicitly for clarity and reasonable speed
            m_j = view(M, :, j)
            q_tm_j = view(QtM, :, j)
            
            # m_star_j = m_j - Q * q_tm_j
            # This allocation inside loop is bad.
            # But Q is n x k. k is small.
            
            # Compute dot products directly?
            # M*_j' M*_j = (M_j - Q Q' M_j)' (M_j - Q Q' M_j)
            # = M_j' M_j - M_j' Q Q' M_j - ...
            # = M_j' M_j - (Q' M_j)' (Q' M_j)
            
            # M*_j' y* = (M_j - Q Q' M_j)' (y - Q Q' y)
            # = M_j' y - M_j' Q Q' y - ...
            # = M_j' y - (Q' M_j)' (Q' y)
            
            mj_mj = dot(m_j, m_j)
            qtm_qtm = dot(q_tm_j, q_tm_j)
            denom = mj_mj - qtm_qtm
            
            if denom < 1e-8
                new_p_values[j] = 1.0
                continue
            end
            
            mj_y = dot(m_j, y)
            qtm_qty = dot(q_tm_j, Qty)
            num = mj_y - qtm_qty
            
            beta = num / denom
            
            # RSS
            # RSS = |y* - beta * M*|^2
            # = |y*|^2 - 2 beta y*' M* + beta^2 |M*|^2
            # = y*' y* - 2 beta num + beta^2 denom
            # = y*' y* - beta * num (since beta = num/denom -> beta*denom = num)
            
            y_star_sq = dot(y_star, y_star)
            rss = y_star_sq - beta * num
            
            if rss < 0; rss = 0.0; end
            
            sigma2 = rss / dof
            
            if sigma2 < 1e-10
                new_p_values[j] = 1.0
                continue
            end
            
            se = sqrt(sigma2 / denom)
            t_stat = beta / se
            
            # P-value
            if abs(t_stat) > 30.0
                new_p_values[j] = 0.0
            else
                new_p_values[j] = 2.0 * ccdf(TDist(dof), abs(t_stat))
            end
        end
        
        p_values = copy(new_p_values)
        
        # 2. Select QTNs (Binning)
        # Divide genome into bins (e.g. LD blocks).
        # For this implementation, we use a simple window-based selection.
        # Pick most significant SNP in each window if p < threshold.
        
        window_size = 100 # SNPs
        threshold = 0.01 / m # Bonferroni-ish
        
        new_qtns = Int[]
        
        # Iterate windows
        for w_start in 1:window_size:m
            w_end = min(w_start + window_size - 1, m)
            window_idx = w_start:w_end
            
            min_p, min_idx_rel = findmin(p_values[window_idx])
            min_idx = window_idx[min_idx_rel]
            
            if min_p < threshold
                push!(new_qtns, min_idx)
            end
        end
        
        # Convergence check
        if Set(new_qtns) == Set(qtns)
            break
        end
        
        qtns = new_qtns
    end
    
    return (p_values=p_values, qtns=qtns)
end


