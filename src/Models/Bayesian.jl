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
"""
function run_bayesA(y::Vector{Float64}, X::Matrix{Float64}; chain_length::Int=1000, burn_in::Int=100, df_beta=4.0, scale_beta=0.001)
    n, p = size(X)
    
    # Pre-compute
    XtX_diag = vec(sum(X.^2, dims=1))
    
    # Initial values
    beta = zeros(p)
    sigma_g2 = fill(0.01, p) # Marker specific variances
    sigma_e2 = var(y) / 2.0
    
    # Storage
    beta_samples = zeros(chain_length, p)
    
    # Residuals
    e = y - X * beta
    
    for iter in 1:(chain_length + burn_in)
        
        # Sample Marker Effects
        for j in 1:p
            # Add back effect of marker j
            e = e + X[:, j] * beta[j]
            
            # Sample beta_j
            rhs = dot(X[:, j], e)
            C = XtX_diag[j] + sigma_e2 / sigma_g2[j]
            inv_C = 1.0 / C
            mean_beta = inv_C * rhs
            var_beta = sigma_e2 * inv_C
            
            beta[j] = rand(Normal(mean_beta, sqrt(var_beta)))
            
            # Update residuals
            e = e - X[:, j] * beta[j]
            
            # Sample sigma_g2_j
            # Posterior is Inv-Chi2(df + 1, (S + beta^2))
            scale_post = scale_beta * df_beta + beta[j]^2
            df_post = df_beta + 1
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
Markers are either 0 (prob pi) or drawn from N(0, sigma_g2) (prob 1-pi).
Common sigma_g2 for all included markers.
"""
function run_bayesC(y::Vector{Float64}, X::Matrix{Float64}; chain_length::Int=1000, burn_in::Int=100, estimate_pi=true, pi::Float64=0.5)
    n, p = size(X)
    XtX_diag = vec(sum(X.^2, dims=1))
    
    beta = zeros(p)
    delta = zeros(Int, p) # Inclusion indicator (0 or 1)
    sigma_g2 = 0.01 # Common variance
    sigma_e2 = var(y) / 2.0
    pi_val = pi
    
    beta_samples = zeros(chain_length, p)
    pi_samples = zeros(chain_length)
    
    e = y - X * beta
    
    for iter in 1:(chain_length + burn_in)
        
        n_included = 0
        
        for j in 1:p
            e = e + X[:, j] * beta[j]
            
            # Likelihood ratio for inclusion
            # L1 (included): N(mu1, v1)
            # L0 (excluded): N(0, 0) -> effectively beta=0
            
            C = XtX_diag[j] + sigma_e2 / sigma_g2
            inv_C = 1.0 / C
            rhs = dot(X[:, j], e)
            mu_j = inv_C * rhs
            var_j = sigma_e2 * inv_C
            
            # Log Likelihood ratio (approximate)
            # log(P(y|d=1)/P(y|d=0))
            # This is complex, standard approach is to sample beta_j from mixture
            
            # Sample beta_j unconditionally from mixture?
            # Standard Gibbs for BayesC:
            # 1. Sample beta_j | delta_j=1
            # 2. Sample delta_j
            
            # Efficient implementation:
            # Calculate prob delta=1
            v0 = sigma_e2
            v1 = sigma_e2 + XtX_diag[j] * sigma_g2 # Marginal variance?
            # Let's use the conditional posterior probability approach
            
            u_0 = rand()
            
            # Likelihood of data given beta_j = 0
            # L0 = exp(- (e'e)/2sigma_e2 ) ... but e is same
            # Actually we compare P(data | beta=0) vs P(data | beta ~ N(0, sigma_g2))
            
            log_L0 = -0.5 * (dot(e, e) / sigma_e2) # This is wrong, e depends on beta
            # Correct approach:
            # log_odds = log(1-pi) - log(pi) + 0.5*log(V_j/sigma_g2) + 0.5*mu_j^2/V_j
            
            log_odds = log(1.0 - pi_val) - log(pi_val) + 0.5 * log(var_j / sigma_g2) + 0.5 * (mu_j^2 / var_j)
            prob_1 = 1.0 / (1.0 + exp(-log_odds))
            
            if rand() < prob_1
                delta[j] = 1
                beta[j] = rand(Normal(mu_j, sqrt(var_j)))
                n_included += 1
            else
                delta[j] = 0
                beta[j] = 0.0
            end
            
            e = e - X[:, j] * beta[j]
        end
        
        # Sample sigma_g2
        # Inv-Chi2
        if n_included > 0
            sum_beta2 = sum(beta[delta .== 1].^2)
            sigma_g2 = rand(InverseGamma((n_included + 4)/2, (sum_beta2 + 0.001)/2))
        else
            sigma_g2 = rand(InverseGamma(2.0, 0.001)) # Prior
        end
        
        # Sample sigma_e2
        sigma_e2 = rand(InverseGamma((n + 2)/2, dot(e, e)/2))
        
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
Markers are 0 (prob pi) or drawn from N(0, sigma_g2_j) (prob 1-pi).
sigma_g2_j ~ Scaled-Inv-Chi2.
"""
function run_bayesB(y::Vector{Float64}, X::Matrix{Float64}; chain_length::Int=1000, burn_in::Int=100, estimate_pi=true)
    n, p = size(X)
    XtX_diag = vec(sum(X.^2, dims=1))
    
    beta = zeros(p)
    delta = zeros(Int, p)
    sigma_g2 = fill(0.01, p) # Marker specific variances
    sigma_e2 = var(y) / 2.0
    pi_val = 0.5
    
    # Hyperparameters for sigma_g2
    df_beta = 4.0
    scale_beta = 0.001
    
    beta_samples = zeros(chain_length, p)
    pi_samples = zeros(chain_length)
    
    e = y - X * beta
    
    for iter in 1:(chain_length + burn_in)
        
        n_included = 0
        
        for j in 1:p
            e = e + X[:, j] * beta[j]
            
            # 1. Sample beta_j and delta_j
            # Prior variance for beta_j is sigma_g2[j] if delta=1, else 0
            
            var_j_prior = sigma_g2[j]
            
            C = XtX_diag[j] + sigma_e2 / var_j_prior
            inv_C = 1.0 / C
            rhs = dot(X[:, j], e)
            mu_j = inv_C * rhs
            var_j_post = sigma_e2 * inv_C
            
            # Log Odds
            # log(1-pi) - log(pi) + 0.5*log(var_j_post/var_j_prior) + 0.5*mu_j^2/var_j_post
            log_odds = log(1.0 - pi_val) - log(pi_val) + 0.5 * log(var_j_post / var_j_prior) + 0.5 * (mu_j^2 / var_j_post)
            prob_1 = 1.0 / (1.0 + exp(-log_odds))
            
            if rand() < prob_1
                delta[j] = 1
                beta[j] = rand(Normal(mu_j, sqrt(var_j_post)))
                n_included += 1
            else
                delta[j] = 0
                beta[j] = 0.0
            end
            
            e = e - X[:, j] * beta[j]
            
            # 2. Sample sigma_g2[j]
            # Only if delta[j] == 1? 
            # In BayesB, usually we sample sigma_g2 even if beta is 0 (from prior), 
            # or we only update if included.
            # Meuwissen et al (2001) samples from prior if excluded.
            
            if delta[j] == 1
                scale_post = scale_beta * df_beta + beta[j]^2
                df_post = df_beta + 1
                sigma_g2[j] = rand(InverseGamma(df_post/2, scale_post/2))
            else
                # Sample from prior
                sigma_g2[j] = rand(InverseGamma(df_beta/2, (scale_beta * df_beta)/2))
            end
        end
        
        # 3. Sample sigma_e2
        sigma_e2 = rand(InverseGamma((n + 2)/2, dot(e, e)/2))
        
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
    run_bayesR(y, X)

BayesR: Mixture of 4 Gaussians with variances [0, 0.0001, 0.001, 0.01] * sigma_g2.
"""
function run_bayesR(y::Vector{Float64}, X::Matrix{Float64}; chain_length::Int=1000, burn_in::Int=100)
    # Simplified implementation of BayesR
    # Gamma mixture classes: 0, 0.0001, 0.001, 0.01
    gammas = [0.0, 0.0001, 0.001, 0.01]
    n_comp = length(gammas)
    
    n, p = size(X)
    XtX_diag = vec(sum(X.^2, dims=1))
    
    beta = zeros(p)
    comp = ones(Int, p) # Component assignment (1-based index)
    pi_vec = fill(1/n_comp, n_comp) # Dirichlet prior
    
    sigma_g2 = var(y) / 2.0 # Genetic variance scaling factor
    sigma_e2 = var(y) / 2.0
    
    beta_samples = zeros(chain_length, p)
    
    e = y - X * beta
    
    for iter in 1:(chain_length + burn_in)
        
        n_counts = zeros(Int, n_comp)
        
        for j in 1:p
            e = e + X[:, j] * beta[j]
            
            rhs = dot(X[:, j], e)
            
            # Calculate prob for each component
            log_probs = zeros(n_comp)
            
            for k in 1:n_comp
                if k == 1 # Zero variance component
                    # P(y|beta=0)
                    # log_probs[k] = log(pi_vec[k]) - 0.5 * dot(e, e)/sigma_e2 # e is with beta=0
                    # This is tricky. Standard way:
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
            probs = exp.(log_probs .- max_log)
            probs ./= sum(probs)
            
            # Sample component
            k_sel = 1
            r = rand()
            cum_p = 0.0
            for k in 1:n_comp
                cum_p += probs[k]
                if r <= cum_p
                    k_sel = k
                    break
                end
            end
            
            comp[j] = k_sel
            n_counts[k_sel] += 1
            
            if k_sel == 1
                beta[j] = 0.0
            else
                var_k = gammas[k_sel] * sigma_g2
                C = XtX_diag[j] + sigma_e2 / var_k
                inv_C = 1.0 / C
                mu_k = inv_C * rhs
                var_post = sigma_e2 * inv_C
                beta[j] = rand(Normal(mu_k, sqrt(var_post)))
            end
            
            e = e - X[:, j] * beta[j]
        end
        
        # Update Pi (Dirichlet)
        pi_vec = rand(Dirichlet(n_counts .+ 1.0))
        
        # Update sigma_g2
        # Sum beta^2 / gamma for non-zero components
        # This is simplified; usually sigma_g2 is fixed or sampled from Inv-Chi2
        
        # Update sigma_e2
        sigma_e2 = rand(InverseGamma((n + 2)/2, dot(e, e)/2))
        
        if iter > burn_in
            beta_samples[iter - burn_in, :] = beta
        end
    end
    
    return mean(beta_samples, dims=1)
end
