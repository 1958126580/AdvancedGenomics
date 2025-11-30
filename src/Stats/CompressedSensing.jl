"""
    CompressedSensing.jl

Compressed Sensing / Sparse Regression module.
Implements Lasso and Elastic Net using Coordinate Descent.
Objective: 1/(2n) ||y - Xb||^2 + lambda * (alpha * ||b||_1 + (1-alpha)/2 * ||b||_2^2)
"""

using LinearAlgebra
using Statistics

"""
    lasso_cd(y, X; lambda=0.1, max_iter=1000, tol=1e-6)

Lasso Regression via Coordinate Descent.
alpha = 1.0 (Pure Lasso).
"""
function lasso_cd(y::Vector{Float64}, X::Matrix{Float64}; lambda::Float64=0.1, max_iter::Int=1000, tol::Float64=1e-6)
    return elastic_net_cd(y, X; lambda=lambda, alpha=1.0, max_iter=max_iter, tol=tol)
end

"""
    elastic_net_cd(y, X; lambda=0.1, alpha=0.5, max_iter=1000, tol=1e-6)

Elastic Net Regression via Coordinate Descent.
alpha: Mixing parameter (1=Lasso, 0=Ridge).
"""
function elastic_net_cd(y::Vector{Float64}, X::Matrix{Float64}; lambda::Float64=0.1, alpha::Float64=0.5, max_iter::Int=1000, tol::Float64=1e-6)
    n, p = size(X)
    
    # Standardize X and y?
    # Usually assumed centered/scaled.
    # We will assume X is centered/scaled for simplicity or handle intercept.
    # Let's assume centered y and X.
    
    beta = zeros(p)
    resid = copy(y) # Residual r = y - Xb
    
    # Pre-compute sum squares of columns
    # If standardized, this is n-1 or n.
    X_sq = vec(sum(X.^2, dims=1))
    
    for iter in 1:max_iter
        max_diff = 0.0
        
        for j in 1:p
            # Partial residual without predictor j
            # r_j = r + X_j * beta_j
            # We update r in place, so we add back contribution
            
            # Efficient update:
            # z_j = x_j' * r + beta_j * (x_j' x_j)
            # But r already contains -X_j*beta_j? No, r = y - sum(Xk bk).
            # So r_without_j = r + X_j * beta_j
            
            # Dot product x_j' * r
            dot_xr = dot(X[:, j], resid)
            
            # z_j = dot_xr + X_sq[j] * beta[j]
            z_j = dot_xr + X_sq[j] * beta[j]
            
            # Soft Thresholding
            # S(z, gamma) = sign(z) * max(|z| - gamma, 0)
            # gamma = n * lambda * alpha
            gamma = n * lambda * alpha
            
            if abs(z_j) <= gamma
                beta_new = 0.0
            else
                beta_new = (z_j - sign(z_j) * gamma) / (X_sq[j] + n * lambda * (1.0 - alpha))
            end
            
            diff = abs(beta_new - beta[j])
            max_diff = max(max_diff, diff)
            
            # Update residual
            if diff > 0
                delta = beta_new - beta[j]
                resid .-= X[:, j] .* delta
                beta[j] = beta_new
            end
        end
        
        if max_diff < tol
            break
        end
    end
    
    return beta
end
