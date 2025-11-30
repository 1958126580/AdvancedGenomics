"""
    LMM.jl

Linear Mixed Model implementation using Gibbs Sampling (MCMC).
Optimized using Eigen-decomposition of K (Kang et al., 2008).
Solves: y = Xb + u + e
Var(u) = K * sigma_g2
Var(e) = I * sigma_e2
"""

using LinearAlgebra
using Statistics
using Distributions

"""
    run_lmm(y, X, K; chain_length=1000, burn_in=100)

Runs a Bayesian Linear Mixed Model.
Returns posterior means and samples.
"""
function run_lmm(y::Vector{Float64}, X::Matrix{Float64}, K::Matrix{Float64}; chain_length::Int=1000, burn_in::Int=100)
    n = length(y)
    p = size(X, 2)
    
    # 1. Diagonalize K
    # K = U S U'
    eig = eigen(Symmetric(K))
    S = eig.values
    U_mat = eig.vectors
    
    # Transform data
    y_star = U_mat' * y
    X_star = U_mat' * X
    
    # Initial values
    beta = zeros(p)
    u_star = zeros(n) 
    sigma_g2 = var(y) / 2.0
    sigma_e2 = var(y) / 2.0
    
    # Storage
    beta_samples = zeros(chain_length, p)
    u_star_samples = zeros(chain_length, n)
    sigma_g2_samples = zeros(chain_length)
    sigma_e2_samples = zeros(chain_length)
    
    XtX_star = X_star' * X_star
    XtX_star_inv = inv(XtX_star + 1e-10I)
    Xty_star = X_star' * y_star
    L_XtX_inv = cholesky(Symmetric(XtX_star_inv)).L
    
    for iter in 1:(chain_length + burn_in)
        
        # 1. Sample Fixed Effects Beta
        # y_corr = y* - u*
        # V_b = sigma_e2 * (X*'X*)^-1
        # mu_b = V_b * (X*' * (y* - u*) / sigma_e2)
        #      = (X*'X*)^-1 * (X*'y* - X*'u*)
        
        Xtu_star = X_star' * u_star
        mu_b = XtX_star_inv * (Xty_star - Xtu_star)
        
        # Sample beta
        # beta ~ N(mu_b, V_b)
        # beta = mu_b + chol(V_b) * z
        # chol(V_b) = sqrt(sigma_e2) * chol((X*'X*)^-1)
        
        beta = mu_b + sqrt(sigma_e2) * L_XtX_inv * randn(p)
        
        # 2. Sample Transformed Random Effects u*
        # u*_i ~ N(mu_i, var_i)
        # Likelihood: y*_i ~ N(x*_i beta + u*_i, sigma_e2)
        # Prior: u*_i ~ N(0, S_i * sigma_g2)
        
        resid_fixed = y_star - X_star * beta
        
        for i in 1:n
            if S[i] < 1e-8
                u_star[i] = 0.0
            else
                prec_u = 1.0 / (S[i] * sigma_g2)
                prec_e = 1.0 / sigma_e2
                prec_post = prec_u + prec_e
                var_post = 1.0 / prec_post
                
                # Mean
                # (y*_i - x*_i beta) * prec_e
                mu_post = var_post * (resid_fixed[i] * prec_e)
                
                u_star[i] = rand(Normal(mu_post, sqrt(var_post)))
            end
        end
        
        # 3. Sample Variance Components
        # sigma_g2 ~ Inv-Chi2
        # Scale = sum( u*_i^2 / S_i )
        # DF = n
        
        # Only use non-zero eigenvalues for DF?
        # Technically u*_i is defined for all, but prior variance is 0 if S_i=0.
        # If S_i=0, u*_i must be 0.
        
        valid_idx = S .> 1e-8
        n_valid = sum(valid_idx)
        scale_g = sum((u_star[valid_idx].^2) ./ S[valid_idx])
        
        # Posterior DF = PriorDF + n_valid
        # Prior DF = -2? Flat prior on log sigma?
        # Let's use weak prior: InvGamma(0.001, 0.001)
        # shape = alpha + n/2
        # scale = beta + scale_g/2
        
        alpha_prior = 0.001
        beta_prior = 0.001
        
        sigma_g2 = rand(InverseGamma(alpha_prior + n_valid/2, beta_prior + scale_g/2))
        
        # sigma_e2
        e_star = resid_fixed - u_star
        scale_e = dot(e_star, e_star)
        
        sigma_e2 = rand(InverseGamma(alpha_prior + n/2, beta_prior + scale_e/2))
        
        if iter > burn_in
            idx = iter - burn_in
            beta_samples[idx, :] = beta
            u_star_samples[idx, :] = u_star
            sigma_g2_samples[idx] = sigma_g2
            sigma_e2_samples[idx] = sigma_e2
        end
    end
    
    # Post-processing
    beta_mean = vec(mean(beta_samples, dims=1))
    sigma_g2_mean = mean(sigma_g2_samples)
    sigma_e2_mean = mean(sigma_e2_samples)
    
    # Transform u back
    # u = U * u*
    # We can average u* first then transform, or transform samples then average.
    # Linear, so equivalent.
    u_star_mean = vec(mean(u_star_samples, dims=1))
    u_mean = U_mat * u_star_mean
    
    return (
        beta = beta_mean,
        u = u_mean,
        sigma_g2 = sigma_g2_mean,
        sigma_e2 = sigma_e2_mean,
        samples = (
            beta = beta_samples,
            sigma_g2 = sigma_g2_samples,
            sigma_e2 = sigma_e2_samples
            # u_samples omitted to save memory unless requested
        )
    )
end
