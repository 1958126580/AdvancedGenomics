"""
    Validation.jl

Cross-Validation utilities.
"""

using Random

"""
    cross_validation(model_func, y, X, k=5)

K-fold Cross-Validation.
"""
function cross_validation(model_func, y::Vector{Float64}, X::Matrix{Float64}, k::Int=5)
    n = length(y)
    indices = shuffle(1:n)
    fold_size = div(n, k)
    
    correlations = zeros(k)
    mses = zeros(k)
    
    for i in 1:k
        val_idx = indices[((i-1)*fold_size + 1):(i*fold_size)]
        train_idx = setdiff(indices, val_idx)
        
        y_train = y[train_idx]
        X_train = X[train_idx, :]
        
        y_val = y[val_idx]
        X_val = X[val_idx, :]
        
        # Train
        beta_hat = model_func(y_train, X_train)
        
        # Predict
        y_pred = X_val * beta_hat' # Assuming beta_hat is 1xp
        
        # Evaluate
        correlations[i] = cor(y_val, vec(y_pred))
        mses[i] = mean((y_val .- vec(y_pred)).^2)
    end
    
    return (mean_cor=mean(correlations), mean_mse=mean(mses))
end
