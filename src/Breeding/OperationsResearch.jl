"""
    OperationsResearch.jl

Operations Research methods for Breeding.
Optimal Contribution Selection (OCS).
"""

using Statistics
using LinearAlgebra


"""
    optimal_contribution_selection(EBV, A; target_inbreeding=0.05, n_offspring=100)

Determines optimal contributions (c) of parents to maximize Genetic Gain (c'EBV)
subject to Inbreeding Constraint (c'Ac <= 2*target_inbreeding) and Sum(c) = 1.
Uses Penalty Method + Simulated Annealing.
"""
function optimal_contribution_selection(EBV::Vector{Float64}, A::Matrix{Float64}; target_inbreeding::Float64=0.05, n_offspring::Int=100)
    n = length(EBV)
    
    # Objective: Maximize c'EBV -> Minimize -c'EBV
    # Constraints:
    # 1. Sum(c) = 1
    # 2. c >= 0
    # 3. 0.5 * c'Ac <= target_inbreeding (Group Coancestry)
    
    # Penalty Function
    function penalty_objective(c_raw)
        # Transform unbounded c_raw to simplex (Softmax)
        # c_i = exp(x_i) / sum(exp(x))
        c = exp.(c_raw) ./ sum(exp.(c_raw))
        
        gain = dot(c, EBV)
        coancestry = 0.5 * (c' * A * c)
        
        # Penalties
        # Constraint: coancestry <= target
        pen_inbreeding = max(0.0, coancestry - target_inbreeding)^2 * 10000.0
        
        # We want to MAX gain, so MIN -gain
        return -gain + pen_inbreeding
    end
    
    # Initial guess (equal contribution)
    x0 = zeros(n) # exp(0) -> 1, equal
    
    # Optimize using SA
    res = simulated_annealing(penalty_objective, x0, max_iter=2000, T_init=10.0)
    
    # Convert back to c
    c_opt = exp.(res.minimizer) ./ sum(exp.(res.minimizer))
    
    # Convert to number of offspring
    n_off = round.(Int, c_opt .* n_offspring)
    
    return (contributions=c_opt, offspring_counts=n_off, expected_gain=dot(c_opt, EBV), group_coancestry=0.5*(c_opt'*A*c_opt))
end
