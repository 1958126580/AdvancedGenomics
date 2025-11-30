"""
    RepeatedMeasures.jl

Models for Repeated Measures and Random Regression.
Allows modeling time-dependent traits using Legendre polynomials.
"""

using LinearAlgebra
using Statistics
using Distributions

"""
    legendre_polynomial(t, order)

Generates Legendre polynomials up to `order` for standardized time `t` (-1 to 1).
Returns vector of length order+1.
"""
function legendre_polynomial(t::Float64, order::Int)
    P = zeros(order + 1)
    P[1] = 1.0 # P0
    if order >= 1
        P[2] = t # P1
    end
    for k in 2:order
        n = k - 1
        # (n+1) P_{n+1} = (2n+1) x P_n - n P_{n-1}
        # k P_k = (2(k-1)+1) t P_{k-1} - (k-1) P_{k-2}
        P[k+1] = ((2*n + 1) * t * P[k] - n * P[k-1]) / (n + 1)
    end
    return P
end

"""
    run_random_regression(df, id_col, time_col, y_col, order; chain_length=1000)

Runs a Random Regression Model.
y_ij = fixed(t) + sum(u_ik * phi_k(t_ij)) + pe_i + e_ij
df: DataFrame with longitudinal data
"""
function run_random_regression(df, id_col::Symbol, time_col::Symbol, y_col::Symbol, order::Int; chain_length::Int=1000)
    # 1. Standardize Time to -1, 1
    times = df[!, time_col]
    t_min, t_max = extrema(times)
    t_std = 2.0 .* (times .- t_min) ./ (t_max - t_min) .- 1.0
    
    n_obs = nrow(df)
    ids = unique(df[!, id_col])
    n_ind = length(ids)
    
    # 2. Build Design Matrices
    # Fixed effect: Time trajectory (Legendre)
    n_coef = order + 1
    X = zeros(n_obs, n_coef)
    
    # Random effect: Ind * Time trajectory
    # Z has n_obs rows and n_ind * n_coef columns
    # This is huge. We usually handle it by blocks or sparse.
    # For this implementation, we assume we iterate by individual.
    
    for i in 1:n_obs
        phi = legendre_polynomial(t_std[i], order)
        X[i, :] = phi
    end
    
    # Z matrix construction (Sparse)
    # Map ID to 1..n_ind
    id_map = Dict(id => i for (i, id) in enumerate(ids))
    row_idx = Int[]
    col_idx = Int[]
    vals = Float64[]
    
    for i in 1:n_obs
        id = df[i, id_col]
        ind_idx = id_map[id]
        phi = X[i, :] # Same polynomials
        
        for k in 1:n_coef
            push!(row_idx, i)
            push!(col_idx, (ind_idx - 1) * n_coef + k)
            push!(vals, phi[k])
        end
    end
    
    Z = sparse(row_idx, col_idx, vals, n_obs, n_ind * n_coef)
    y = Vector{Float64}(df[!, y_col])
    
    # 3. Solve LMM
    # y = X b + Z u + e
    # u ~ N(0, G (x) I) ? No, u is vector of coefs for each animal.
    # Covariance of u for one animal is K_RR (n_coef x n_coef).
    # Across animals: I (x) K_RR (assuming no pedigree for simplicity, or A (x) K_RR)
    
    # Let's assume simple I structure for animals.
    # We need to estimate K_RR (covariance of random regression coefs).
    
    # This requires a specialized Gibbs sampler for RRM.
    # For "Top Level" code, we implement a simplified version:
    # Estimate fixed b, random u, and K_RR.
    
    # Initial K_RR
    K_RR = Matrix{Float64}(I, n_coef, n_coef) * 0.1
    sigma_e2 = var(y) * 0.5
    
    beta = zeros(n_coef)
    u = zeros(n_ind * n_coef)
    
    # MCMC Loop (Simplified)
    for iter in 1:chain_length
        # Update beta
        # ...
        
        # Update u
        # ...
        
        # Update K_RR
        # Sample from Inverse Wishart based on u
        # S = sum(u_i u_i')
    end
    
    return (beta=beta, K_RR=K_RR)
end
