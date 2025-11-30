"""
    ModernStats.jl

Modern Statistical Methods for Genomics.
Bayes Factors, Wavelet Analysis, Conformal Prediction.
"""

using Statistics
using Distributions
using LinearAlgebra
using Random

# --- Bayes Factors ---

"""
    calculate_bayes_factor(posterior_samples; prior_dist=Normal(0, 1), at=0.0)

Computes Bayes Factor (BF_10) for the hypothesis H1: theta != at vs H0: theta = at.
Uses Savage-Dickey Density Ratio: BF_10 = p(D | H1) / p(D | H0) = p(theta=at | H1) / p(theta=at | D, H1) ?
Actually Savage-Dickey: BF_01 = p(theta=at | D) / p(theta=at | Prior).
So BF_10 = p(theta=at | Prior) / p(theta=at | Posterior).

posterior_samples: Vector of MCMC samples for the parameter.
prior_dist: Prior distribution object (e.g., Normal(0,1)).
at: The point null value (usually 0).
"""
function calculate_bayes_factor(posterior_samples::Vector{Float64}; prior_dist::Distribution=Normal(0, 1), at::Float64=0.0)
    # Density at 'at' for Prior
    prior_dens = pdf(prior_dist, at)
    
    # Density at 'at' for Posterior
    # We need to estimate the density from samples.
    # Kernel Density Estimation (KDE) or simple histogram approximation.
    # For "National Top Level", let's use a simple Gaussian Kernel approximation.
    
    bw = 1.06 * std(posterior_samples) * length(posterior_samples)^(-1/5) # Silverman's rule
    if bw == 0
        bw = 1e-6
    end
    
    # KDE at 'at'
    post_dens = mean([pdf(Normal(s, bw), at) for s in posterior_samples])
    
    if post_dens == 0
        return Inf # Strong evidence against H0 (if prior > 0)
    end
    
    # BF_10 = Prior / Posterior
    bf_10 = prior_dens / post_dens
    
    # Interpretation:
    # BF_10 > 1: Evidence for H0 (theta = at) ???
    # Wait. Savage-Dickey gives BF_01 (Evidence FOR H0).
    # BF_01 = Posterior(at) / Prior(at).
    # If Posterior at 0 is high, H0 is supported.
    # We want BF_10 (Evidence FOR H1, i.e., effect exists).
    # BF_10 = 1 / BF_01 = Prior(at) / Posterior(at).
    
    return bf_10
end

# --- Wavelet Analysis ---

"""
    wavelet_denoising(signal; level=2, threshold_type=:soft)

Denoises a 1D signal using Discrete Wavelet Transform (Haar).
Simple implementation for demo.
"""
function wavelet_denoising(signal::Vector{Float64}; level::Int=2, threshold_type::Symbol=:soft)
    n = length(signal)
    # Pad to power of 2
    n_pow2 = nextpow(2, n)
    x = vcat(signal, zeros(n_pow2 - n))
    
    # Forward Transform (Haar)
    coeffs = copy(x)
    
    # Multilevel
    current_len = n_pow2
    
    # Store details
    # In-place Haar:
    # [A, D] -> [A_new, D_new, D]
    
    # We'll do a simple recursive implementation or iterative.
    
    for l in 1:level
        half = current_len ÷ 2
        temp = zeros(current_len)
        for i in 1:half
            # Avg (Approximation)
            temp[i] = (coeffs[2*i-1] + coeffs[2*i]) / sqrt(2)
            # Diff (Detail)
            temp[half + i] = (coeffs[2*i-1] - coeffs[2*i]) / sqrt(2)
        end
        coeffs[1:current_len] = temp
        current_len = half
    end
    
    # Thresholding (on Details)
    # Details are from current_len+1 to end of that level
    # Universal threshold: sigma * sqrt(2*log(n))
    sigma = median(abs.(coeffs[(n_pow2 ÷ 2 + 1):end])) / 0.6745 # MAD estimate of noise
    thresh = sigma * sqrt(2 * log(n_pow2))
    
    # Apply threshold to all detail coefficients
    # Details are indices > current_len (final approximation size)
    # Actually, we should threshold all detail levels.
    # The final approximation is 1:current_len.
    # All others are details.
    
    for i in (current_len + 1):n_pow2
        val = coeffs[i]
        if threshold_type == :soft
            coeffs[i] = sign(val) * max(0.0, abs(val) - thresh)
        else
            if abs(val) < thresh
                coeffs[i] = 0.0
            end
        end
    end
    
    # Inverse Transform
    current_len = n_pow2 >> (level - 1) # Start from smallest approximation?
    # Wait, loop backwards
    # Level 1 produced size/2 approx and size/2 detail.
    # Level 2 produced size/4 approx and size/4 detail.
    # Final approx is size/(2^level).
    
    # We reconstruct from level 'level' down to 1.
    
    # Current approx size
    curr_size = n_pow2 >> level
    
    for l in 1:level
        # Reconstruct curr_size * 2
        half = curr_size
        full = curr_size * 2
        temp = zeros(full)
        
        for i in 1:half
            avg = coeffs[i]
            diff = coeffs[half + i]
            
            # x1 = (avg + diff) / sqrt(2)
            # x2 = (avg - diff) / sqrt(2)
            
            temp[2*i-1] = (avg + diff) / sqrt(2)
            temp[2*i] = (avg - diff) / sqrt(2)
        end
        coeffs[1:full] = temp
        curr_size = full
    end
    
    return coeffs[1:n]
end

# --- Conformal Prediction ---

"""
    conformal_prediction(X_train, y_train, X_test; alpha=0.1)

Split Conformal Prediction for Regression.
Returns prediction intervals (lower, upper).
Uses a simple Ridge Regression as the base model.
"""
function conformal_prediction(X_train::Matrix{Float64}, y_train::Vector{Float64}, X_test::Matrix{Float64}; alpha::Float64=0.1)
    n = size(X_train, 1)
    
    # Split calibration set
    n_cal = floor(Int, 0.5 * n)
    idx = randperm(n)
    idx_train = idx[1:(n-n_cal)]
    idx_cal = idx[(n-n_cal+1):end]
    
    X_tr = X_train[idx_train, :]
    y_tr = y_train[idx_train]
    
    X_cal = X_train[idx_cal, :]
    y_cal = y_train[idx_cal]
    
    # Train base model (Ridge)
    # beta = (X'X + lambda I)^-1 X'y
    lambda = 1.0
    beta = (X_tr' * X_tr + lambda * I) \ (X_tr' * y_tr)
    
    # Predict on Calibration
    y_cal_pred = X_cal * beta
    
    # Compute non-conformity scores (Absolute Residuals)
    scores = abs.(y_cal .- y_cal_pred)
    
    # Quantile
    # q = (1 - alpha) * (1 + 1/n_cal) quantile
    q_val = quantile(scores, min(1.0, (1.0 - alpha) * (1.0 + 1.0/n_cal)))
    
    # Predict on Test
    y_test_pred = X_test * beta
    
    lower = y_test_pred .- q_val
    upper = y_test_pred .+ q_val
    
    return (pred=y_test_pred, lower=lower, upper=upper)
end
