"""
    INLA.jl

Integrated Nested Laplace Approximation (INLA) module.
Provides fast Bayesian inference for Latent Gaussian Models (LMM).
Approximates posterior distributions using Laplace approximation instead of MCMC.
"""

using LinearAlgebra
using Statistics
using Optim # We need an optimizer, but we'll implement a simple one or use Optim if available. 
# Since Optim is not in standard lib, we'll implement a simple Grid Search or Nelder-Mead for 1-2 parameters.
# For LMM with 1 variance ratio (lambda), Grid Search is efficient.

"""
    run_inla_lmm(y, X, Z, K_inv; grid_size=100)

Runs INLA for a Linear Mixed Model.
y = Xb + Zu + e
Var(u) = K * sigma_g2
Var(e) = I * sigma_e2
lambda = sigma_e2 / sigma_g2

We approximate p(lambda | y) and then integrate.
"""
function run_inla_lmm(y::Vector{Float64}, X::AbstractMatrix, Z::AbstractMatrix, K_inv::AbstractMatrix; grid_size::Int=100)
    n = length(y)
    p = size(X, 2)
    q = size(Z, 2)
    
    # Pre-compute fixed matrices
    XtX = X' * X
    XtZ = X' * Z
    ZtX = Z' * X
    ZtZ = Z' * Z
    Xty = X' * y
    Zty = Z' * y
    
    # Grid search for log(lambda)
    # lambda = sigma_e2 / sigma_g2
    # Range: 1e-4 to 1e4
    log_lambdas = range(-4.0, 4.0, length=grid_size)
    log_posteriors = zeros(grid_size)
    
    # Store results for integration
    results = []
    
    max_log_post = -Inf
    
    for (i, log_lambda) in enumerate(log_lambdas)
        lambda = 10.0^log_lambda
        
        # 1. Build MME LHS for this lambda
        # [XtX  XtZ]
        # [ZtX  ZtZ + K_inv * lambda]
        
        # Efficient block construction
        if issparse(K_inv)
            lower_right = ZtZ + K_inv .* lambda
            LHS = [sparse(XtX) sparse(XtZ);
                   sparse(ZtX) lower_right]
        else
            lower_right = ZtZ + K_inv .* lambda
            LHS = [XtX XtZ;
                   ZtX lower_right]
        end
        
        RHS = [Xty; Zty]
        
        # 2. Solve MME (Mode of latent field x | theta)
        # x_hat = LHS \ RHS
        # Factorize LHS
        # Use Cholesky if positive definite (should be)
        # F = cholesky(Symmetric(LHS))
        # But LHS might not be perfectly symmetric due to float errors, force Symmetric
        
        try
            # For dense matrices
            if !issparse(LHS)
                LHS_sym = Symmetric(LHS)
                F = cholesky(LHS_sym)
                x_hat = F \ RHS
                log_det_Q = logdet(F) * 2.0 # log|Q|
            else
                # Sparse Cholesky
                F = cholesky(Symmetric(LHS))
                x_hat = F \ RHS
                log_det_Q = logdet(F)
            end
            
            beta_hat = x_hat[1:p]
            u_hat = x_hat[p+1:end]
            
            # 3. Compute Laplace Approximation of Marginal Likelihood
            # log p(y | lambda) approx -0.5 * log|Q| - 0.5 * (y - ...)'(y - ...) / sigma_e2 ... 
            # Actually, we need to profile out sigma_e2 or integrate it.
            # Standard REML log-likelihood profile for lambda:
            # log L_R(lambda) = -0.5 * log|V| - 0.5 * log|X'V^-1X| - 0.5 * (y'Py)
            # This is equivalent to INLA approximation.
            
            # Let's use the MME relationship:
            # log|V| + log|X'V^-1X| = log|C| - log|K^-1 * lambda| ? No.
            # Relationship: log|C| = log|X'X| + log|Z'Z + K^-1 lambda| ... complicated.
            
            # Easier: Use the REML form directly via MME factorization.
            # log L = -0.5 * (log|C| + n*log(sigma_e2) + y'Py/sigma_e2 + const) + 0.5 * log|K^-1 * lambda|
            
            # Estimate sigma_e2 for this lambda
            # resid = y - X*beta - Z*u
            # SSE = resid' * resid + lambda * u' * K_inv * u
            # sigma_e2_hat = SSE / (n - p)
            
            resid = y - X * beta_hat - Z * u_hat
            quad_u = dot(u_hat, K_inv * u_hat)
            SSE = dot(resid, resid) + lambda * quad_u
            
            sigma_e2_hat = SSE / (n - p)
            
            # Log Posterior (assuming flat prior on log(lambda))
            # log p(lambda|y) propto log L_R(lambda)
            # log|C| = log_det_Q
            # log|K_inv| is constant
            # term: -0.5 * log_det_Q - 0.5 * (n-p) * log(sigma_e2_hat)
            # Correction for K_inv * lambda determinant:
            # The MME includes K_inv*lambda.
            # We need to be careful with determinants.
            
            # Simplified REML objective:
            # log L = -0.5 * log_det_Q - 0.5 * (n-p) * log(SSE) + 0.5 * q * log(lambda)
            
            val = -0.5 * log_det_Q - 0.5 * (n - p) * log(SSE) + 0.5 * q * log(lambda)
            
            log_posteriors[i] = val
            if val > max_log_post
                max_log_post = val
            end
            
            push!(results, (lambda=lambda, beta=beta_hat, u=u_hat, sigma_e2=sigma_e2_hat, log_p=val))
            
        catch e
            # Cholesky failed (not PD)
            log_posteriors[i] = -Inf
        end
    end
    
    # 4. Integrate (Normalize posterior)
    # exp(log_p - max)
    weights = exp.(log_posteriors .- max_log_post)
    weights ./= sum(weights)
    
    # 5. Compute Bayesian Model Averaging
    final_beta = zeros(p)
    final_u = zeros(q)
    final_sigma_e2 = 0.0
    final_sigma_g2 = 0.0
    
    valid_count = 0
    
    for (i, res) in enumerate(results)
        w = weights[i]
        if w > 1e-10
            final_beta .+= w .* res.beta
            final_u .+= w .* res.u
            final_sigma_e2 += w * res.sigma_e2
            final_sigma_g2 += w * (res.sigma_e2 / res.lambda)
            valid_count += 1
        end
    end
    
    return (beta=final_beta, u=final_u, sigma_e2=final_sigma_e2, sigma_g2=final_sigma_g2)
end
