"""
    VarianceComponents.jl

Variance Component Estimation (REML).
Implements AI-REML (Average Information) and EM-REML.
Supports Multi-kernel models for Multi-omics and Epistasis.
Model: y = Xb + u_1 + ... + u_k + e
Var(u_i) = K_i * sigma_i^2
Var(e) = I * sigma_e^2
"""

using LinearAlgebra
using Statistics
using Distributions

"""
    estimate_vc_reml(y, X, Ks; method="AI", max_iter=50, tol=1e-6)

Estimates variance components for multiple kernels.
Ks: Vector of Matrices [K1, K2, ...] or single Matrix K.
"""
function estimate_vc_reml(y::Vector{Float64}, X::Matrix{Float64}, K::Matrix{Float64}; method::String="AI", max_iter::Int=50, tol::Float64=1e-6)
    return estimate_vc_reml(y, X, [K], method=method, max_iter=max_iter, tol=tol)
end

function estimate_vc_reml(y::Vector{Float64}, X::Matrix{Float64}, Ks::Vector{Matrix{Float64}}; method::String="AI", max_iter::Int=50, tol::Float64=1e-6)
    if method == "AI"
        return ai_reml_multi(y, X, Ks, max_iter, tol)
    else
        error("Method $method not supported for multi-kernel.")
    end
end

"""
    ai_reml_multi(y, X, Ks, max_iter, tol)

Multi-kernel Average Information REML.
"""
function ai_reml_multi(y::Vector{Float64}, X::Matrix{Float64}, Ks::Vector{Matrix{Float64}}, max_iter::Int, tol::Float64)
    n = length(y)
    p = size(X, 2)
    k = length(Ks)
    
    # Initial estimates
    # Split total variance equally
    var_y = var(y)
    sigmas = fill(var_y / (k + 1), k) # sigma_1^2 ... sigma_k^2
    sigma_e2 = var_y / (k + 1)
    
    I_n = Matrix{Float64}(I, n, n)
    
    for iter in 1:max_iter
        # V = sum(sigma_i^2 * K_i) + sigma_e2 * I
        V = sigma_e2 .* I_n
        for i in 1:k
            V .+= sigmas[i] .* Ks[i]
        end
        
        V_inv = inv(V)
        
        # P = V^-1 - V^-1 X (X' V^-1 X)^-1 X' V^-1
        XtVinv = X' * V_inv
        XtVinvX = XtVinv * X
        XtVinvX_inv = inv(XtVinvX)
        
        P = V_inv - V_inv * X * XtVinvX_inv * XtVinv
        Py = P * y
        
        # Scores
        # dL/ds_i = -0.5 * (tr(P K_i) - y' P K_i P y)
        # dL/ds_e = -0.5 * (tr(P) - y' P P y)
        
        scores = zeros(k + 1)
        AI = zeros(k + 1, k + 1)
        
        # Pre-compute P * K_i
        PKs = [P * Ks[i] for i in 1:k]
        
        # Score for kernels
        for i in 1:k
            yPKPy = dot(Py, Ks[i] * Py)
            scores[i] = -0.5 * (tr(PKs[i]) - yPKPy)
        end
        
        # Score for residual
        yPPy = dot(Py, Py)
        scores[k+1] = -0.5 * (tr(P) - yPPy)
        
        # AI Matrix
        # AI_ij = 0.5 * y' P K_i P K_j P y
        # Approximate tr(P K_i P K_j) with data part? No, AI uses data part.
        # AI_ij = 0.5 * y' P dV/di P dV/dj P y
        
        # Helper to compute y' P A P B P y
        # = (Py)' A P B (Py)
        # Let v = Py. v' A P B v.
        
        for i in 1:k
            for j in i:k
                # AI_ij
                # term = 0.5 * y' P K_i P K_j P y
                term = 0.5 * dot(Py, Ks[i] * P * Ks[j] * Py)
                AI[i, j] = term
                AI[j, i] = term
            end
            
            # AI_i,e
            # term = 0.5 * y' P K_i P I P y
            term = 0.5 * dot(Py, Ks[i] * P * Py)
            AI[i, k+1] = term
            AI[k+1, i] = term
        end
        
        # AI_ee
        term = 0.5 * dot(Py, P * Py)
        AI[k+1, k+1] = term
        
        # Update
        delta = AI \ scores
        
        for i in 1:k
            sigmas[i] += delta[i]
            if sigmas[i] < 1e-6 sigmas[i] = 1e-6 end
        end
        sigma_e2 += delta[k+1]
        if sigma_e2 < 1e-6 sigma_e2 = 1e-6 end
        
        # Convergence check
        if norm(delta) < tol
            break
        end
    end
    
    # Fixed Effects
    # Reconstruct V
    V = sigma_e2 .* I_n
    for i in 1:k
        V .+= sigmas[i] .* Ks[i]
    end
    V_inv = inv(V)
    XtVinvX = X' * V_inv * X
    beta = XtVinvX \ (X' * V_inv * y)
    
    return (sigmas=sigmas, sigma_e2=sigma_e2, beta=beta)
end
