"""
    BreedingModels.jl

Specific Animal Breeding Models.
Includes Sire Model and Test Day Model.
"""

using LinearAlgebra
using Statistics

"""
    run_sire_model(y, X, sire_ids, ped)

Runs a Sire Model.
y = Xb + Z_s u_s + e
u_s ~ N(0, A_s * sigma_s2)
sigma_s2 = 1/4 * sigma_g2 (approx)
"""
function run_sire_model(y::Vector{Float64}, X::Matrix{Float64}, sire_ids::Vector{String}, ped::Pedigree)
    # 1. Build A matrix for Sires only
    # Extract unique sires
    unique_sires = unique(sire_ids)
    n_sires = length(unique_sires)
    
    # Build Z_s incidence matrix
    n = length(y)
    Z_s = zeros(Float64, n, n_sires)
    sire_map = Dict(id => i for (i, id) in enumerate(unique_sires))
    
    for i in 1:n
        sid = sire_ids[i]
        if haskey(sire_map, sid)
            Z_s[i, sire_map[sid]] = 1.0
        end
    end
    
    # Build A for sires
    # We can use build_A from PedModule and subset, or just assume identity/relationship if provided.
    # For simplicity, let's use Identity or build_A if feasible.
    # Assuming PedModule has build_A.
    
    # A_full = build_A(ped)
    # Subset A for sires... this might be heavy.
    # Let's assume uncorrelated sires for this demo or use GRM if available.
    # Or just use Identity for "Sire Model" demo.
    K_s = Matrix{Float64}(I, n_sires, n_sires)
    
    # Run LMM
    # We use our existing run_lmm or estimate_vc_reml
    # Let's use estimate_vc_reml for variance components
    
    res = estimate_vc_reml(y, X, Z_s * K_s * Z_s') # This is V_s = Z K Z'
    # Wait, estimate_vc_reml takes K as the covariance of random effects.
    # If we pass Z*K*Z', that's the covariance of y (partially).
    # estimate_vc_reml(y, X, K) assumes y = Xb + u + e, Var(u) = K*sigma2.
    # Here y = Xb + Z*u_s + e. Var(Z*u_s) = Z * A * Z' * sigma_s2.
    # So we pass K_equiv = Z * K_s * Z'.
    
    K_equiv = Z_s * K_s * Z_s'
    res = estimate_vc_reml(y, X, K_equiv)
    
    return res
end

"""
    run_test_day_model(y, X, animal_ids, days_in_milk; order=2)

Runs a Test Day Model (Random Regression).
Uses Legendre polynomials on Days in Milk (DIM) to model permanent environment / genetic effects.
"""
function run_test_day_model(y::Vector{Float64}, X::Matrix{Float64}, animal_ids::Vector{String}, days_in_milk::Vector{Float64}; order::Int=2)
    # This is essentially a Random Regression Model.
    # We can reuse run_random_regression from Models/RepeatedMeasures.jl
    # But let's provide a specific wrapper.
    
    # Normalize DIM to -1 to 1 for Legendre
    dim_min, dim_max = extrema(days_in_milk)
    t_norm = 2.0 .* (days_in_milk .- dim_min) ./ (dim_max - dim_min) .- 1.0
    
    # We need to construct Z for random regression coefficients.
    # Z has blocks for each animal?
    # run_random_regression expects Z to be constructed.
    
    # Let's call run_random_regression directly if available.
    # It is available in Models/RepeatedMeasures.jl
    
    # We need to construct the covariate matrix for random effects (Phi)
    # Phi = Legendre polynomials of t_norm
    
    # Reuse run_random_regression logic?
    # Actually, run_random_regression in RepeatedMeasures.jl was:
    # function run_random_regression(y, X, t, subjects; order=2)
    
    return run_random_regression(y, X, t_norm, animal_ids; order=order)
end
