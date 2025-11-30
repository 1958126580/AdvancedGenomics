"""
    Multivariate.jl

Multivariate Linear Mixed Models (Multi-trait LMM).
Optimized using Row-wise Block Gibbs Sampling for Random Effects.
Solves: Y = XB + U + E
Var(u_i) = Sigma_G (t x t)
Cov(u_i, u_j) = K_ij * Sigma_G
Var(e_i) = Sigma_E (t x t)
"""

using LinearAlgebra
using Statistics
using Distributions
using Random

"""
    run_multitrait_lmm(Y, X, K; chain_length=1000, burn_in=100)

Runs a multi-trait Bayesian LMM.
"""
function run_multitrait_lmm(Y::Matrix{Float64}, X::Matrix{Float64}, K::Matrix{Float64}; chain_length::Int=1000, burn_in::Int=100)
    n, t = size(Y)
    p = size(X, 2)
    
    # Priors / Initial Values
    Sigma_G = Matrix{Float64}(I, t, t)
    Sigma_E = Matrix{Float64}(I, t, t)
    
    B = zeros(p, t)
    U = zeros(n, t)
    
    Sigma_G_samples = zeros(chain_length, t, t)
    
    K_inv = inv(K + 1e-6I)
    
    # Pre-compute for B sampling
    XtX = X' * X
    XtX_inv = inv(XtX + 1e-8I)
    
    for iter in 1:(chain_length + burn_in)
        
        # 1. Sample B (Fixed Effects)
        # Conditional on U, Sigma_E
        # vec(B) ~ N(vec(B_hat), Sigma_E (x) (X'X)^-1)
        Sigma_E_inv = inv(Sigma_E)
        Y_corr = Y - U
        B_hat = XtX_inv * (X' * Y_corr)
        
        # Sample matrix normal
        # B = B_hat + E1 * chol(Sigma_E)'
        # where E1 rows are correlated by (X'X)^-1?
        # Actually: vec(B) = vec(B_hat) + chol(Sigma_E (x) (X'X)^-1) * z
        # = vec(B_hat) + (chol(Sigma_E) (x) chol((X'X)^-1)) * z
        # B = B_hat + chol((X'X)^-1) * Z * chol(Sigma_E)'
        
        L_XtX = cholesky(Symmetric(XtX_inv)).L
        L_Sigma_E = cholesky(Symmetric(Sigma_E + 1e-8I)).L
        Z_mat = randn(p, t)
        B = B_hat + L_XtX * Z_mat * L_Sigma_E'
        
        # 2. Sample U (Random Effects) - Row-wise Block Gibbs
        # Sample u_i (t x 1) conditional on u_{-i}, B, Sigma_G, Sigma_E
        # This avoids solving nt x nt system.
        # P(u_i | ...) propto P(y_i | u_i) * P(u_i | u_{-i})
        
        # Likelihood: y_i ~ N(x_i B + u_i, Sigma_E)
        # Prior: u ~ N(0, K (x) Sigma_G)
        # Conditional Prior: u_i | u_{-i}
        # This is standard Gaussian conditional.
        # u_i | u_{-i} ~ N( mu_prior_i, V_prior_i )
        # mu_prior_i = - sum_{j!=i} K_ij/K_ii * u_j  (if K inverse is used?)
        # Actually easier with K inverse:
        # P(U) propto exp(-0.5 * tr(U' K^-1 U Sigma_G^-1))
        # Part involving u_i:
        # tr(U' K^-1 U Sigma_G^-1) = sum_k sum_l (K^-1)_kl u_k' Sigma_G^-1 u_l
        # = (K^-1)_ii u_i' Sigma_G^-1 u_i + 2 sum_{j!=i} (K^-1)_ij u_i' Sigma_G^-1 u_j
        
        # So Prior for u_i is proportional to:
        # exp( -0.5 * [ u_i' ((K^-1)_ii Sigma_G^-1) u_i + 2 u_i' (Sigma_G^-1 sum_{j!=i} (K^-1)_ij u_j) ] )
        # This is Kernel of N(mu, V)
        # V_prior_inv = (K^-1)_ii * Sigma_G^-1
        # V_prior_inv * mu_prior = - Sigma_G^-1 * sum_{j!=i} (K^-1)_ij u_j
        # mu_prior = - (1/(K^-1)_ii) * sum_{j!=i} (K^-1)_ij u_j
        
        Sigma_G_inv = inv(Sigma_G)
        
        Resid = Y - X * B
        
        for i in 1:n
            # Prior parameters
            k_ii = K_inv[i, i]
            sum_k_u = zeros(t)
            for j in 1:n
                if i != j
                    sum_k_u += K_inv[i, j] * U[j, :]
                end
            end
            
            # V_prior_inv = k_ii * Sigma_G_inv
            # Prior precision * mu = - Sigma_G_inv * sum_k_u
            
            # Likelihood: y_i ~ N(u_i, Sigma_E) (centered)
            # Precision_lik = Sigma_E_inv
            # Precision_lik * y_i = Sigma_E_inv * Resid[i, :]
            
            # Posterior Precision
            # P_post = k_ii * Sigma_G_inv + Sigma_E_inv
            # P_post * mu_post = - Sigma_G_inv * sum_k_u + Sigma_E_inv * Resid[i, :]
            
            P_post = k_ii .* Sigma_G_inv .+ Sigma_E_inv
            rhs = -Sigma_G_inv * sum_k_u + Sigma_E_inv * Resid[i, :]
            
            V_post = inv(P_post)
            mu_post = V_post * rhs
            
            # Sample u_i
            L_post = cholesky(Symmetric(V_post + 1e-10I)).L
            U[i, :] = mu_post + L_post * randn(t)
        end
        
        # 3. Sample Sigma_G
        # IW(nu + n, S + U' K^-1 U)
        S_G = U' * K_inv * U
        df_G = t + n
        # Ensure symmetry for InverseWishart
        Psi_G = S_G + I
        Psi_G = (Psi_G + Psi_G') ./ 2.0
        Sigma_G = rand(InverseWishart(df_G, Psi_G))
        
        # 4. Sample Sigma_E
        # IW(nu + n, S + E'E)
        E = Resid - U
        S_E = E' * E
        df_E = t + n
        Psi_E = S_E + I
        Psi_E = (Psi_E + Psi_E') ./ 2.0
        Sigma_E = rand(InverseWishart(df_E, Psi_E))
        
        if iter > burn_in
            idx = iter - burn_in
            Sigma_G_samples[idx, :, :] = Sigma_G
        end
    end
    
    return (
        Sigma_G = dropdims(mean(Sigma_G_samples, dims=1), dims=1),
        B = B
    )
end
