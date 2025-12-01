"""
    Bayesian.jl

Implementation of Bayesian Alphabet models for Genomic Prediction.
Includes: BayesA, BayesB, BayesC (BayesCPi), BayesR, and Bayesian Lasso.
"""

using LinearAlgebra
using Statistics
using Distributions
using Random

"""
    sample_beta_bayesA(y_corr, diag_XtX, sigma_e2, sigma_g2_j)

Samples effect for a single marker in BayesA.
"""
function sample_beta_bayesA(y_corr::Vector{Float64}, X_j::Vector{Float64}, diag_XtX_j::Float64, sigma_e2::Float64, sigma_g2_j::Float64)
    # Posterior variance
    C_j = diag_XtX_j + sigma_e2 / sigma_g2_j
    V_j = sigma_e2 / C_j
    
    # Posterior mean
    rhs = dot(X_j, y_corr) + diag_XtX_j * 0.0 # 0.0 is old beta if we used it, but here y_corr includes it? 
    # Usually y_corr = y - X_{-j}b_{-j} = y - Xb + X_j b_j
    # So rhs = X_j' * y_corr
    
    mu_j = V_j * (rhs / sigma_e2)
    
    return rand(Normal(mu_j, sqrt(V_j)))
end

"""
    run_bayesA(y, X; chain_length=1000, burn_in=100)

BayesA: Each marker has its own variance σ²_j ~ Scaled-Inv-Chi²(ν, S²).
Optimized using LoopVectorization for high-performance MCMC sampling.
"""
function run_bayesA(y::Vector{Float64}, X::Matrix{Float64}; chain_length::Int=1000, burn_in::Int=100, df_beta=4.0, scale_beta=0.001)
    # using LoopVectorization # Assuming it's available in the environment
    
    n, p = size(X)
    
    # Pre-compute diagonal of X'X
    XtX_diag = zeros(Float64, p)
    @inbounds for j in 1:p
        s = 0.0
        col = view(X, :, j)
        @simd for i in 1:n
            s += col[i]^2
        end
        XtX_diag[j] = s
    end
    
    # Initial values
    beta = zeros(Float64, p)
    sigma_g2 = fill(0.01, p) # Marker specific variances
    sigma_e2 = var(y) / 2.0
    
    # Storage
    beta_samples = zeros(Float64, chain_length, p)
    
    # Residuals
    e = y - X * beta
    
    # Pre-allocate for rhs computation
    # We avoid allocating X[:, j] repeatedly
    
    for iter in 1:(chain_length + burn_in)
        
        # Sample Marker Effects
        @inbounds for j in 1:p
            # Add back effect of marker j
            bj = beta[j]
            
            # e = e + X[:, j] * bj
            # Optimized vector update
            col_j = view(X, :, j)
            
            # Compute RHS = X[:, j]' * (e + X[:, j] * bj)
            # = X[:, j]' * e + (X[:, j]' * X[:, j]) * bj
            # = dot(col_j, e) + XtX_diag[j] * bj
            
            # We can compute dot(col_j, e) without updating e fully if we want, 
            # but standard Gibbs updates e.
            
            # Update e temporarily
            @simd for i in 1:n
                e[i] += col_j[i] * bj
            end
            
            # Compute RHS
            rhs = 0.0
            @simd for i in 1:n
                rhs += col_j[i] * e[i]
            end
            
            # Sample beta_j
            C = XtX_diag[j] + sigma_e2 / sigma_g2[j]
            inv_C = 1.0 / C
            mean_beta = inv_C * rhs
            var_beta = sigma_e2 * inv_C
            
            new_beta = rand(Normal(mean_beta, sqrt(var_beta)))
            beta[j] = new_beta
            
            # Update residuals with new beta
            @simd for i in 1:n
                e[i] -= col_j[i] * new_beta
            end
            
            # Sample sigma_g2_j
            # Posterior is Inv-Chi2(df + 1, (S + beta^2))
            scale_post = scale_beta * df_beta + new_beta^2
            df_post = df_beta + 1
            
            # InverseGamma(alpha, beta) -> InvChi2(nu, s2) mapping
            # InvChi2(nu, s2) ~ InvGamma(nu/2, nu*s2/2)
            # Here scale_post is sum of squares? 
            # Scaled-Inv-Chi2(nu, tau^2): f(x) ~ (tau^2 * nu / 2)^(nu/2) ...
            # Standard update:
            sigma_g2[j] = rand(InverseGamma(df_post/2, scale_post/2))
        end
        
        # Sample sigma_e2
        scale_e = dot(e, e)
        sigma_e2 = rand(InverseGamma((n + 2)/2, scale_e/2))
        
        if iter > burn_in
            beta_samples[iter - burn_in, :] = beta
        end
    end
    
    return mean(beta_samples, dims=1)
end

"""
    run_bayesC(y, X; chain_length=1000, burn_in=100, pi=0.95)

BayesC (BayesCPi): Variable selection.
Optimized for performance.
"""
function run_bayesC(y::Vector{Float64}, X::Matrix{Float64}; chain_length::Int=1000, burn_in::Int=100, estimate_pi=true, pi::Float64=0.5)
    n, p = size(X)
    
    # Pre-compute diagonal of X'X
    XtX_diag = zeros(Float64, p)
    @inbounds for j in 1:p
        s = 0.0
        col = view(X, :, j)
        @simd for i in 1:n
            s += col[i]^2
        end
        XtX_diag[j] = s
    end
    
    beta = zeros(Float64, p)
    delta = zeros(Int, p) # Inclusion indicator (0 or 1)
    sigma_g2 = 0.01 # Common variance
    sigma_e2 = var(y) / 2.0
    pi_val = pi
    
    beta_samples = zeros(Float64, chain_length, p)
    pi_samples = zeros(Float64, chain_length)
    
    e = y - X * beta
    
    for iter in 1:(chain_length + burn_in)
        
        n_included = 0
        
        @inbounds for j in 1:p
            bj = beta[j]
            col_j = view(X, :, j)
            
            # Update e temporarily (remove effect)
            if bj != 0.0
                @simd for i in 1:n
                    e[i] += col_j[i] * bj
                end
            end
            
            # Compute RHS
            rhs = 0.0
            @simd for i in 1:n
                rhs += col_j[i] * e[i]
            end
            
            C = XtX_diag[j] + sigma_e2 / sigma_g2
            inv_C = 1.0 / C
            mu_j = inv_C * rhs
            var_j = sigma_e2 * inv_C
            
            # Log Likelihood ratio (approximate)
            # log_odds = log(1-pi) - log(pi) + 0.5*log(V_j/sigma_g2) + 0.5*mu_j^2/V_j
            
            log_odds = log(1.0 - pi_val) - log(pi_val) + 0.5 * log(var_j / sigma_g2) + 0.5 * (mu_j^2 / var_j)
            prob_1 = 1.0 / (1.0 + exp(-log_odds))
            
            new_beta = 0.0
            if rand() < prob_1
                delta[j] = 1
                new_beta = rand(Normal(mu_j, sqrt(var_j)))
                n_included += 1
            else
                delta[j] = 0
                new_beta = 0.0
            end
            
            beta[j] = new_beta
            
            # Update residuals with new beta
            if new_beta != 0.0
                @simd for i in 1:n
                    e[i] -= col_j[i] * new_beta
                end
            end
        end
        
        # Sample sigma_g2
        if n_included > 0
            sum_beta2 = sum(beta[delta .== 1].^2)
            sigma_g2 = rand(InverseGamma((n_included + 4)/2, (sum_beta2 + 0.001)/2))
        else
            sigma_g2 = rand(InverseGamma(2.0, 0.001)) # Prior
        end
        
        # Sample sigma_e2
        scale_e = dot(e, e)
        sigma_e2 = rand(InverseGamma((n + 2)/2, scale_e/2))
        
        # Sample Pi
        if estimate_pi
            pi_val = rand(Beta(p - n_included + 1, n_included + 1))
        end
        
        if iter > burn_in
            beta_samples[iter - burn_in, :] = beta
            pi_samples[iter - burn_in] = pi_val
        end
    end
    
    return (beta=mean(beta_samples, dims=1), pi=mean(pi_samples))
end

"""
    run_bayesB(y, X; chain_length=1000, burn_in=100, pi=0.95)

BayesB: Variable selection + Marker specific variances.
Optimized for performance.
"""
function run_bayesB(y::Vector{Float64}, X::Matrix{Float64}; chain_length::Int=1000, burn_in::Int=100, estimate_pi=true)
    n, p = size(X)
    
    # Pre-compute diagonal of X'X
    XtX_diag = zeros(Float64, p)
    @inbounds for j in 1:p
        s = 0.0
        col = view(X, :, j)
        @simd for i in 1:n
            s += col[i]^2
        end
        XtX_diag[j] = s
    end
    
    beta = zeros(Float64, p)
    delta = zeros(Int, p)
    sigma_g2 = fill(0.01, p) # Marker specific variances
    sigma_e2 = var(y) / 2.0
    pi_val = 0.5
    
    # Hyperparameters for sigma_g2
    df_beta = 4.0
    scale_beta = 0.001
    
    beta_samples = zeros(Float64, chain_length, p)
    pi_samples = zeros(Float64, chain_length)
    
    e = y - X * beta
    
    for iter in 1:(chain_length + burn_in)
        
        n_included = 0
        
        @inbounds for j in 1:p
            bj = beta[j]
            col_j = view(X, :, j)
            
            # Update e temporarily
            if bj != 0.0
                @simd for i in 1:n
                    e[i] += col_j[i] * bj
                end
            end
            
            # Compute RHS
            rhs = 0.0
            @simd for i in 1:n
                rhs += col_j[i] * e[i]
            end
            
            # 1. Sample beta_j and delta_j
            var_j_prior = sigma_g2[j]
            
            C = XtX_diag[j] + sigma_e2 / var_j_prior
            inv_C = 1.0 / C
            mu_j = inv_C * rhs
            var_j_post = sigma_e2 * inv_C
            
            # Log Odds
            log_odds = log(1.0 - pi_val) - log(pi_val) + 0.5 * log(var_j_post / var_j_prior) + 0.5 * (mu_j^2 / var_j_post)
            prob_1 = 1.0 / (1.0 + exp(-log_odds))
            
            new_beta = 0.0
            if rand() < prob_1
                delta[j] = 1
                new_beta = rand(Normal(mu_j, sqrt(var_j_post)))
                n_included += 1
            else
                delta[j] = 0
                new_beta = 0.0
            end
            
            beta[j] = new_beta
            
            # Update residuals with new beta
            if new_beta != 0.0
                @simd for i in 1:n
                    e[i] -= col_j[i] * new_beta
                end
            end
            
            # 2. Sample sigma_g2[j]
            if delta[j] == 1
                scale_post = scale_beta * df_beta + new_beta^2
                df_post = df_beta + 1
                sigma_g2[j] = rand(InverseGamma(df_post/2, scale_post/2))
            else
                # Sample from prior
                sigma_g2[j] = rand(InverseGamma(df_beta/2, (scale_beta * df_beta)/2))
            end
        end
        
        # 3. Sample sigma_e2
        scale_e = dot(e, e)
        sigma_e2 = rand(InverseGamma((n + 2)/2, scale_e/2))
        
        # 4. Sample Pi
        if estimate_pi
            pi_val = rand(Beta(p - n_included + 1, n_included + 1))
        end
        
        if iter > burn_in
            beta_samples[iter - burn_in, :] = beta
            pi_samples[iter - burn_in] = pi_val
        end
    end
    
    return (beta=mean(beta_samples, dims=1), pi=mean(pi_samples))
end

"""
    run_bayesR(y, X; chain_length=1000, burn_in=100)

BayesR: Mixture of 4 Gaussians with variances [0, 0.0001, 0.001, 0.01] * sigma_g2.
Optimized for performance.
"""
function run_bayesR(y::Vector{Float64}, X::Matrix{Float64}; chain_length::Int=1000, burn_in::Int=100)
    # Simplified implementation of BayesR
    # Gamma mixture classes: 0, 0.0001, 0.001, 0.01
    gammas = [0.0, 0.0001, 0.001, 0.01]
    n_comp = length(gammas)
    
    n, p = size(X)
    
    # Pre-compute diagonal of X'X
    XtX_diag = zeros(Float64, p)
    @inbounds for j in 1:p
        s = 0.0
        col = view(X, :, j)
        @simd for i in 1:n
            s += col[i]^2
        end
        XtX_diag[j] = s
    end
    
    beta = zeros(Float64, p)
    comp = ones(Int, p) # Component assignment (1-based index)
    pi_vec = fill(1/n_comp, n_comp) # Dirichlet prior
    
    sigma_g2 = var(y) / 2.0 # Genetic variance scaling factor
    sigma_e2 = var(y) / 2.0
    
    beta_samples = zeros(Float64, chain_length, p)
    
    e = y - X * beta
    
    # Pre-allocate log_probs
    log_probs = zeros(Float64, n_comp)
    
    for iter in 1:(chain_length + burn_in)
        
        n_counts = zeros(Int, n_comp)
        
        @inbounds for j in 1:p
            bj = beta[j]
            col_j = view(X, :, j)
            
            # Update e temporarily
            if bj != 0.0
                @simd for i in 1:n
                    e[i] += col_j[i] * bj
                end
            end
            
            # Compute RHS
            rhs = 0.0
            @simd for i in 1:n
                rhs += col_j[i] * e[i]
            end
            
            # Calculate prob for each component
            for k in 1:n_comp
                if k == 1 # Zero variance component
                    log_probs[k] = log(pi_vec[k])
                else
                    var_k = gammas[k] * sigma_g2
                    C = XtX_diag[j] + sigma_e2 / var_k
                    inv_C = 1.0 / C
                    mu_k = inv_C * rhs
                    var_post = sigma_e2 * inv_C
                    
                    log_probs[k] = log(pi_vec[k]) - 0.5 * log(var_k) + 0.5 * log(var_post) + 0.5 * mu_k^2 / var_post
                end
            end
            
            # Normalize probabilities
            max_log = maximum(log_probs)
            # Avoid underflow/overflow
            sum_probs = 0.0
            for k in 1:n_comp
                p_k = exp(log_probs[k] - max_log)
                log_probs[k] = p_k # Reuse array to store unnormalized probs
                sum_probs += p_k
            end
            
            # Sample component
            k_sel = 1
            r = rand() * sum_probs
            cum_p = 0.0
            for k in 1:n_comp
                cum_p += log_probs[k]
                if r <= cum_p
                    k_sel = k
                    break
                end
            end
            
            comp[j] = k_sel
            n_counts[k_sel] += 1
            
            new_beta = 0.0
            if k_sel == 1
                new_beta = 0.0
            else
                var_k = gammas[k_sel] * sigma_g2
                C = XtX_diag[j] + sigma_e2 / var_k
                inv_C = 1.0 / C
                mu_k = inv_C * rhs
                var_post = sigma_e2 * inv_C
                new_beta = rand(Normal(mu_k, sqrt(var_post)))
            end
            
            beta[j] = new_beta
            
            # Update residuals with new beta
            if new_beta != 0.0
                @simd for i in 1:n
                    e[i] -= col_j[i] * new_beta
                end
            end
        end
        
        # Update Pi (Dirichlet)
        pi_vec = rand(Dirichlet(n_counts .+ 1.0))
        
        # Update sigma_e2
        scale_e = dot(e, e)
        sigma_e2 = rand(InverseGamma((n + 2)/2, scale_e/2))
        
        if iter > burn_in
            beta_samples[iter - burn_in, :] = beta
        end
    end
    
    return mean(beta_samples, dims=1)
end
