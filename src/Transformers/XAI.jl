"""
    XAI.jl

Explainable AI (XAI) for Genomic Models.
Implements Saliency Maps (Input Gradients).
"""

using Lux
using LinearAlgebra
using Statistics

"""
    saliency_map(model, x, ps, st)

Computes the gradient of the model output with respect to the input `x`.
Since `x` is discrete (indices), we typically look at the gradient w.r.t the embedding vectors.
"""
function saliency_map(model, x, ps, st)
    # x: Input indices (seq_len, batch)
    # Compute gradient of output with respect to input
    
    # For discrete inputs (indices), we compute gradient w.r.t embeddings
    # This gives us the sensitivity of each position
    
    # Forward pass to get baseline output
    y_orig, _ = model(x, ps, st)
    
    # Use Zygote to compute gradients
    # We compute the gradient of the sum of outputs w.r.t. the input
    grads = Zygote.gradient(x -> begin
        y, _ = model(x, ps, st)
        sum(y)
    end, x)
    
    # Extract gradient magnitude for each position
    grad_x = grads[1]
    
    if grad_x === nothing
        # Fallback to occlusion sensitivity if gradients not available
        seq_len = size(x, 1)
        saliency = zeros(seq_len)
        y_val = y_orig[1]
        
        for i in 1:seq_len
            x_occ = copy(x)
            x_occ[i, :] .= 1 # Mask with 1 (neutral value)
            
            y_occ, _ = model(x_occ, ps, st)
            diff = abs(y_val - y_occ[1])
            saliency[i] = diff
        end
        
        return saliency
    else
        # Return absolute gradient values as saliency
        # Average across batch dimension if present
        if ndims(grad_x) > 1
            return vec(mean(abs.(grad_x), dims=2))
        else
            return abs.(grad_x)
        end
    end
end
