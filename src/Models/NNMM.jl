"""
    NNMM.jl

Neural Network Mixed Models (NN-MM).
Integrates a Neural Network into the Mixed Model framework.
y = Xb + NN(G) + Zu + e
The NN output is treated as a random effect or covariate with a prior.
"""

using Lux
using LinearAlgebra
using Statistics
using Random
using Distributions

"""
    run_nnmm(y, G_idx, X; hidden_dim=32, chain_length=1000)

Runs a single-step NN-MM.
In each MCMC iteration, we update the NN weights using HMC or Gradient Descent (Langevin Dynamics),
and then update the other parameters using Gibbs.
"""
function run_nnmm(y::Vector{Float64}, G_idx::Matrix{Int}, X::Matrix{Float64}; hidden_dim::Int=32, chain_length::Int=1000)
    n, p = size(X)
    m = size(G_idx, 1) # SNPs
    
    # Define NN: Input (m SNPs) -> Hidden -> Output (1 scalar)
    # We use an embedding layer for SNPs? Or just dense input?
    # G_idx is m x n (SNPs x Inds).
    # Let's use a simple MLP on the genotype vector.
    
    model = Chain(
        Dense(m, hidden_dim, relu),
        Dense(hidden_dim, 1)
    )
    
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)
    
    # Convert G to Float for Dense layer
    G_float = Float32.(G_idx) # m x n
    
    # MCMC Loop
    # We need to sample posterior of weights `ps`.
    # Likelihood: y ~ N(Xb + NN(G), sigma_e2)
    # Prior on weights: N(0, sigma_w2)
    
    # This requires Hamiltonian Monte Carlo (HMC) or SGLD for weights.
    # For simplicity, we implement a Metropolis-Hastings step for weights.
    
    sigma_e2 = var(y) / 2.0
    beta = zeros(p)
    
    for iter in 1:chain_length
        # 1. Update Beta (Gibbs)
        # y_corr = y - NN(G)
        nn_pred = vec(Lux.apply(model, G_float, ps, st)[1]) # n vector
        y_corr = y - nn_pred
        
        # Standard Bayesian Regression for beta
        XtX = X' * X
        V_b = inv(XtX) * sigma_e2
        mu_b = V_b * (X' * y_corr) / sigma_e2
        beta = rand(MvNormal(mu_b, V_b + 1e-8I))
        
        # 2. Update NN Weights (Metropolis-Hastings)
        # Implement Random Walk MH for neural network parameters
        
        # Helper function to flatten Lux parameters to vector
        function flatten_params(ps)
            vcat([vec(ps[k]) for k in keys(ps)]...)
        end
        
        # Helper function to unflatten vector back to Lux parameters
        function unflatten_params(v, ps_template)
            ps_new = deepcopy(ps_template)
            idx = 1
            for k in keys(ps_template)
                len = length(ps_template[k])
                ps_new[k] = reshape(v[idx:idx+len-1], size(ps_template[k]))
                idx += len
            end
            return ps_new
        end
        
        # Log-likelihood function
        function log_likelihood(ps_current, y, X, beta, G_float, model, st, sigma_e2)
            nn_pred = vec(Lux.apply(model, G_float, ps_current, st)[1])
            residuals = y - X * beta - nn_pred
            return -0.5 * sum(residuals.^2) / sigma_e2 - 0.5 * n * log(2π * sigma_e2)
        end
        
        # Log-prior (Gaussian prior on weights)
        function log_prior(params_vec, sigma_w2=1.0)
            return -0.5 * sum(params_vec.^2) / sigma_w2 - 0.5 * length(params_vec) * log(2π * sigma_w2)
        end
        
        # Flatten current parameters
        params_vec = flatten_params(ps)
        n_params = length(params_vec)
        
        # Proposal: Random walk with adaptive step size
        proposal_sd = 0.01  # Small step size for stability
        params_proposal = params_vec + proposal_sd * randn(n_params)
        
        # Unflatten to Lux format
        ps_proposal = unflatten_params(params_proposal, ps)
        
        # Calculate acceptance ratio
        log_lik_current = log_likelihood(ps, y, X, beta, G_float, model, st, sigma_e2)
        log_lik_proposal = log_likelihood(ps_proposal, y, X, beta, G_float, model, st, sigma_e2)
        
        log_prior_current = log_prior(params_vec)
        log_prior_proposal = log_prior(params_proposal)
        
        log_alpha = (log_lik_proposal + log_prior_proposal) - (log_lik_current + log_prior_current)
        
        # Accept/reject
        if log(rand()) < log_alpha
            ps = ps_proposal  # Accept
        end
        # else: keep current ps (reject)
        
        # 3. Update sigma_e2 (Gibbs)
        nn_pred = vec(Lux.apply(model, G_float, ps, st)[1])
        residuals = y - X * beta - nn_pred
        sigma_e2 = sum(residuals.^2) / n  # Simple update (could use InverseGamma prior)
    end
    
    return (beta=beta, model=model, ps=ps)
end
