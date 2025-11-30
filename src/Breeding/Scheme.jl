"""
    Scheme.jl

Breeding Scheme Design module.
Includes Selection Index and Genetic Gain prediction.
"""

using LinearAlgebra
using Statistics

"""
    selection_index(P, G, w)

Calculates Selection Index weights (b).
b = P^-1 * G * w
P: Phenotypic covariance matrix (n_traits x n_traits)
G: Genetic covariance matrix (n_traits x n_traits)
w: Economic weights (n_traits)
"""
function selection_index(P::Matrix{Float64}, G::Matrix{Float64}, w::Vector{Float64})
    # Hazel's Index
    # I = b'y
    # H = w'g
    # Maximize correlation r_IH
    
    b = P \ (G * w)
    return b
end

"""
    predict_genetic_gain(i, r_TI, sigma_g)

Predicts Genetic Gain (Delta G).
Delta G = i * r_TI * sigma_g
i: Selection intensity
r_TI: Accuracy of selection
sigma_g: Genetic standard deviation
"""
function predict_genetic_gain(i::Float64, r_TI::Float64, sigma_g::Float64)
    return i * r_TI * sigma_g
end

"""
    predict_response_to_selection(b, G, w, i, sigma_I)

Predicts response for each trait in the index.
R = i * G * b / sigma_I
sigma_I = sqrt(b' P b)
"""
function predict_response_to_selection(b::Vector{Float64}, P::Matrix{Float64}, G::Matrix{Float64}, i::Float64)
    sigma_I = sqrt(dot(b, P * b))
    if sigma_I == 0
        return zeros(length(b))
    end
    
    # Response vector (correlated response)
    # R = i * G * b / sigma_I
    R = (i / sigma_I) .* (G * b)
    return R
end
