"""
    ML.jl

Machine Learning algorithms for Genomic Selection.
Random Forest and Gradient Boosting (Native Implementation).
"""

using Statistics
using Random

# --- Decision Tree (Regression) ---

struct Node
    feature::Int
    threshold::Float64
    value::Float64
    left::Union{Node, Nothing}
    right::Union{Node, Nothing}
    is_leaf::Bool
end

function build_tree(X::Matrix{Float64}, y::Vector{Float64}; max_depth::Int=5, min_samples::Int=5, depth::Int=0)
    n, m = size(X)
    
    # Base cases
    if depth >= max_depth || n < min_samples || var(y) == 0
        return Node(0, 0.0, mean(y), nothing, nothing, true)
    end
    
    # Find best split
    best_mse = Inf
    best_feat = 0
    best_thresh = 0.0
    best_left_idx = Int[]
    best_right_idx = Int[]
    
    # Random feature selection (for RF)
    n_feats_try = floor(Int, sqrt(m))
    feats = randperm(m)[1:n_feats_try]
    
    for j in feats
        # Try thresholds
        # Optimization: Quantiles or unique values
        vals = unique(X[:, j])
        if length(vals) < 2 continue end
        
        # Sample thresholds if too many
        if length(vals) > 10
            thresholds = quantile(vals, range(0.1, 0.9, length=10))
        else
            thresholds = (vals[1:end-1] .+ vals[2:end]) ./ 2
        end
        
        for t in thresholds
            left = X[:, j] .<= t
            right = .!left
            
            if sum(left) < 2 || sum(right) < 2 continue end
            
            y_left = y[left]
            y_right = y[right]
            
            mse = sum((y_left .- mean(y_left)).^2) + sum((y_right .- mean(y_right)).^2)
            
            if mse < best_mse
                best_mse = mse
                best_feat = j
                best_thresh = t
                best_left_idx = findall(left)
                best_right_idx = findall(right)
            end
        end
    end
    
    if best_feat == 0
        return Node(0, 0.0, mean(y), nothing, nothing, true)
    end
    
    left_node = build_tree(X[best_left_idx, :], y[best_left_idx], max_depth=max_depth, min_samples=min_samples, depth=depth+1)
    right_node = build_tree(X[best_right_idx, :], y[best_right_idx], max_depth=max_depth, min_samples=min_samples, depth=depth+1)
    
    return Node(best_feat, best_thresh, 0.0, left_node, right_node, false)
end

function predict_tree(node::Node, x::Vector{Float64})
    if node.is_leaf
        return node.value
    end
    
    if x[node.feature] <= node.threshold
        return predict_tree(node.left, x)
    else
        return predict_tree(node.right, x)
    end
end

# --- Random Forest ---

struct RandomForest
    trees::Vector{Node}
end

"""
    random_forest(X, y; n_trees=10, max_depth=5)

Trains a Random Forest Regressor.
"""
function random_forest(X::Matrix{Float64}, y::Vector{Float64}; n_trees::Int=10, max_depth::Int=5)
    n = size(X, 1)
    trees = Node[]
    
    Threads.@threads for i in 1:n_trees
        # Bootstrap sample
        indices = rand(1:n, n)
        X_boot = X[indices, :]
        y_boot = y[indices]
        
        tree = build_tree(X_boot, y_boot, max_depth=max_depth)
        # Lock push?
        # Use a temporary array and merge
    end
    
    # Re-do sequential for safety in demo
    for i in 1:n_trees
        indices = rand(1:n, n)
        X_boot = X[indices, :]
        y_boot = y[indices]
        push!(trees, build_tree(X_boot, y_boot, max_depth=max_depth))
    end
    
    return RandomForest(trees)
end

function predict_rf(rf::RandomForest, X::Matrix{Float64})
    n = size(X, 1)
    preds = zeros(n)
    
    for i in 1:n
        val = 0.0
        for tree in rf.trees
            val += predict_tree(tree, X[i, :])
        end
        preds[i] = val / length(rf.trees)
    end
    return preds
end

# --- Gradient Boosting ---

struct GradientBoosting
    trees::Vector{Node}
    learning_rate::Float64
    base_score::Float64
end

"""
    gradient_boosting(X, y; n_trees=10, max_depth=3, lr=0.1)

Trains a Gradient Boosting Regressor.
"""
function gradient_boosting(X::Matrix{Float64}, y::Vector{Float64}; n_trees::Int=10, max_depth::Int=3, lr::Float64=0.1)
    trees = Node[]
    base_score = mean(y)
    preds = fill(base_score, length(y))
    
    for i in 1:n_trees
        # Compute residuals
        resid = y .- preds
        
        # Fit tree to residuals
        tree = build_tree(X, resid, max_depth=max_depth)
        push!(trees, tree)
        
        # Update predictions
        for j in 1:length(y)
            preds[j] += lr * predict_tree(tree, X[j, :])
        end
    end
    
    return GradientBoosting(trees, lr, base_score)
end

function predict_gbm(gbm::GradientBoosting, X::Matrix{Float64})
    n = size(X, 1)
    preds = fill(gbm.base_score, n)
    
    for i in 1:n
        for tree in gbm.trees
            preds[i] += gbm.learning_rate * predict_tree(tree, X[i, :])
        end
    end
    return preds
end
