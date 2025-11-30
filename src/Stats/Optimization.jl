"""
    Optimization.jl

Advanced Optimization Algorithms for Genomics.
Simulated Annealing, Genetic Algorithm, Ant Colony Optimization.
"""

using Statistics
using Random
using LinearAlgebra

# --- Simulated Annealing ---

"""
    simulated_annealing(objective_func, x0; T_init=100.0, cooling_rate=0.95, max_iter=1000)

Minimizes objective_func(x) using Simulated Annealing.
x0: Initial parameter vector.
"""
function simulated_annealing(objective_func::Function, x0::Vector{Float64}; T_init::Float64=100.0, cooling_rate::Float64=0.95, max_iter::Int=1000)
    current_x = copy(x0)
    current_cost = objective_func(current_x)
    
    best_x = copy(current_x)
    best_cost = current_cost
    
    T = T_init
    
    for i in 1:max_iter
        # Propose neighbor (Gaussian perturbation)
        neighbor_x = current_x .+ randn(length(x0)) .* 0.1
        neighbor_cost = objective_func(neighbor_x)
        
        delta = neighbor_cost - current_cost
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 || rand() < exp(-delta / T)
            current_x = neighbor_x
            current_cost = neighbor_cost
            
            if current_cost < best_cost
                best_x = copy(current_x)
                best_cost = current_cost
            end
        end
        
        T *= cooling_rate
    end
    
    return (minimizer=best_x, minimum=best_cost)
end

# --- Genetic Algorithm (Feature Selection) ---

"""
    genetic_algorithm_select(X, y; pop_size=50, generations=20, mutation_rate=0.01)

Feature Selection using Genetic Algorithm.
Maximizes R^2 of Linear Regression (Ridge).
Returns binary mask of selected features.
"""
function genetic_algorithm_select(X::Matrix{Float64}, y::Vector{Float64}; pop_size::Int=50, generations::Int=20, mutation_rate::Float64=0.01)
    n, m = size(X)
    
    # Initialize population (random binary masks)
    population = [rand(Bool, m) for _ in 1:pop_size]
    
    # Fitness function
    function fitness(mask)
        if sum(mask) == 0 return 0.0 end
        X_sub = X[:, mask]
        # Ridge regression
        # beta = (X'X + I)^-1 X'y
        try
            beta = (X_sub' * X_sub + I) \ (X_sub' * y)
            y_pred = X_sub * beta
            # R2
            ss_tot = sum((y .- mean(y)).^2)
            ss_res = sum((y .- y_pred).^2)
            return 1.0 - ss_res / ss_tot
        catch
            return 0.0
        end
    end
    
    best_mask = population[1]
    best_fit = -Inf
    
    for gen in 1:generations
        scores = [fitness(ind) for ind in population]
        
        # Track best
        max_idx = argmax(scores)
        if scores[max_idx] > best_fit
            best_fit = scores[max_idx]
            best_mask = population[max_idx]
        end
        
        # Selection (Tournament)
        new_pop = Vector{Vector{Bool}}(undef, pop_size)
        for i in 1:pop_size
            # Tournament of 3
            cands = rand(1:pop_size, 3)
            winner = cands[argmax(scores[cands])]
            new_pop[i] = copy(population[winner])
        end
        
        # Crossover (Single point)
        for i in 1:2:pop_size
            if i+1 > pop_size break end
            if rand() < 0.8 # Crossover prob
                pt = rand(1:m)
                p1 = new_pop[i]
                p2 = new_pop[i+1]
                
                new_pop[i] = vcat(p1[1:pt], p2[pt+1:end])
                new_pop[i+1] = vcat(p2[1:pt], p1[pt+1:end])
            end
        end
        
        # Mutation
        for i in 1:pop_size
            for j in 1:m
                if rand() < mutation_rate
                    new_pop[i][j] = !new_pop[i][j]
                end
            end
        end
        
        population = new_pop
    end
    
    return best_mask
end

# --- Ant Colony Optimization (Feature Selection) ---

"""
    ant_colony_select(X, y; n_ants=10, iterations=10, n_features=10)

Feature Selection using ACO.
Selects a fixed number of features `n_features`.
"""
function ant_colony_select(X::Matrix{Float64}, y::Vector{Float64}; n_ants::Int=10, iterations::Int=10, n_features::Int=10)
    n, m = size(X)
    
    # Pheromones on features
    tau = ones(m)
    
    best_mask = zeros(Bool, m)
    best_fit = -Inf
    
    # Fitness (same as GA)
    function fitness(mask)
        if sum(mask) == 0 return 0.0 end
        X_sub = X[:, mask]
        try
            beta = (X_sub' * X_sub + I) \ (X_sub' * y)
            y_pred = X_sub * beta
            ss_tot = sum((y .- mean(y)).^2)
            ss_res = sum((y .- y_pred).^2)
            return 1.0 - ss_res / ss_tot
        catch
            return 0.0
        end
    end
    
    for iter in 1:iterations
        # Ants construct solutions
        solutions = []
        scores = []
        
        for k in 1:n_ants
            # Probabilistic construction
            # P(j) ~ tau[j]
            probs = tau ./ sum(tau)
            # Sample n_features without replacement
            # Weighted sampling
            
            mask = zeros(Bool, m)
            # Simple weighted sampling loop
            available = collect(1:m)
            current_tau = copy(tau)
            
            for _ in 1:n_features
                if isempty(available) break end
                p = current_tau[available]
                if sum(p) == 0 
                    idx = rand(available)
                else
                    p = p ./ sum(p)
                    # Sample
                    r = rand()
                    cum = cumsum(p)
                    sel_idx = findfirst(cum .>= r)
                    idx = available[sel_idx]
                end
                
                mask[idx] = true
                # Remove from available
                filter!(x -> x != idx, available)
            end
            
            fit = fitness(mask)
            push!(solutions, mask)
            push!(scores, fit)
            
            if fit > best_fit
                best_fit = fit
                best_mask = mask
            end
        end
        
        # Pheromone Update
        # Evaporation
        tau *= 0.9
        # Deposit
        for (mask, score) in zip(solutions, scores)
            # Deposit proportional to score?
            # If score > 0
            if score > 0
                tau[mask] .+= score
            end
        end
    end
    
    return best_mask
end
