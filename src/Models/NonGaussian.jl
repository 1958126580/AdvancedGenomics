"""
    NonGaussian.jl

Models for Non-Gaussian traits.
Includes Threshold Models for categorical data (Ordinal/Binary).
"""

using LinearAlgebra
using Statistics
using Distributions


"""
    run_threshold_model(y_cat, X, K; chain_length=1000, burn_in=100)

Threshold Model for Ordinal/Binary Data.
y_cat: Vector of integers (1, 2, ..., C)
Uses data augmentation: y_li ~ N(mu_i, 1)
y_cat_i = k if t_{k-1} < y_li < t_k
"""
function run_threshold_model(y_cat::Vector{Int}, X::Matrix{Float64}, K::Matrix{Float64}; chain_length::Int=1000, burn_in::Int=100)
    n = length(y_cat)
    p = size(X, 2)
    categories = sort(unique(y_cat))
    n_cat = length(categories)
    
    # Thresholds
    # t_0 = -Inf, t_1 = 0 (fixed), t_C = Inf
    # We need to estimate t_2 ... t_{C-1}
    thresholds = zeros(n_cat + 1)
    thresholds[1] = -Inf
    thresholds[2] = 0.0
    thresholds[end] = Inf
    
    # Initialize thresholds for C > 2
    if n_cat > 2
        for k in 3:n_cat
            thresholds[k] = thresholds[k-1] + 1.0
        end
    end
    
    # Latent variable y_l
    y_l = zeros(n)
    # Initialize y_l consistent with y_cat
    for i in 1:n
        c = y_cat[i]
        lo = thresholds[c]
        hi = thresholds[c+1]
        if isinf(lo) lo = -2.0 end
        if isinf(hi) hi = 2.0 end
        y_l[i] = (lo + hi) / 2.0
    end
    
    # Parameters
    beta = zeros(p)
    u = zeros(n)
    sigma_g2 = 1.0
    # sigma_e2 is fixed to 1.0 for identification in threshold models
    
    K_inv = inv(K + 1e-6I)
    
    beta_samples = zeros(chain_length, p)
    
    for iter in 1:(chain_length + burn_in)
        
        # 1. Sample Latent Variable y_l
        # y_li | rest ~ N(x_i'b + u_i, 1) truncated to (t_{c-1}, t_c)
        mu_vec = X * beta + u
        for i in 1:n
            c = y_cat[i]
            lo = thresholds[c]
            hi = thresholds[c+1]
            
            # Sample truncated normal
            # Using simple rejection or inverse CDF if available
            # For speed, just approximate or use Distributions
            
            # d = Normal(mu_vec[i], 1.0)
            # y_l[i] = rand(Truncated(d, lo, hi))
            
            # Simple rejection sampling for demo
            valid = false
            while !valid
                val = randn() + mu_vec[i]
                if val > lo && val < hi
                    y_l[i] = val
                    valid = true
                end
                # Fallback for tight bounds
                if !valid
                     y_l[i] = (max(lo, -10.0) + min(hi, 10.0)) / 2.0 # Placeholder
                     valid = true
                end
            end
        end
        
        # 2. Sample Fixed Effects beta
        # Standard LMM step with y_l as response
        # y_corr = y_l - u
        # beta ~ N((X'X)^-1 X'y_corr, (X'X)^-1)
        XtX = X' * X
        V_b = inv(XtX)
        mu_b = V_b * (X' * (y_l - u))
        beta = rand(MvNormal(mu_b, V_b + 1e-8I))
        
        # 3. Sample Random Effects u
        # LHS = I + K^-1 / sigma_g2
        # RHS = y_l - X*beta
        lambda = 1.0 / sigma_g2
        LHS = I + K_inv * lambda
        RHS = y_l - X * beta
        V_u = inv(LHS)
        mu_u = V_u * RHS
        u = rand(MvNormal(mu_u, V_u + 1e-8I))
        
        # 4. Sample sigma_g2
        scale_g = dot(u, K_inv, u)
        sigma_g2 = rand(InverseGamma(n/2, scale_g/2))
        
        # 5. Update Thresholds (if n_cat > 2)
        if n_cat > 2
            # Metropolis-Hastings for thresholds
            # Uniform prior on ordered thresholds
            for k in 3:n_cat
                # Range is (t_{k-1}, t_{k+1})
                # Also constrained by max(y_l | y=k-1) and min(y_l | y=k)
                
                lo_bound = maximum(y_l[y_cat .== k-1])
                hi_bound = minimum(y_l[y_cat .== k])
                
                # Sample uniform in valid range
                if hi_bound > lo_bound
                    thresholds[k] = rand(Uniform(lo_bound, hi_bound))
                end
            end
        end
        
        if iter > burn_in
            beta_samples[iter - burn_in, :] = beta
        end
    end
    
    return mean(beta_samples, dims=1)
end
