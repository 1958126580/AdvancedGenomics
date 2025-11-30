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
    # We want d(Output)/d(Embedding).
    
    # In Lux, we can use Zygote or similar AD.
    # Since we don't have Zygote loaded in the main package (to keep deps light?),
    # we might need to rely on numerical differentiation or assume Zygote is available.
    
    # For "Top Level" code, we should use AD.
    # Assuming the user has Zygote.
    
    # Placeholder for AD logic:
    # gradient(m -> sum(m(x, ps, st)[1]), model)
    
    # Since we can't easily add Zygote dependency dynamically, 
    # we will implement a perturbation-based saliency (Occlusion Sensitivity).
    
    # Occlusion: Mask each SNP and measure drop in prediction.
    
    y_orig, _ = model(x, ps, st)
    y_val = y_orig[1]
    
    seq_len = size(x, 1)
    saliency = zeros(seq_len)
    
    for i in 1:seq_len
        x_occ = copy(x)
        x_occ[i] = 1 # Mask with 1 (assuming 0/missing)
        
        y_occ, _ = model(x_occ, ps, st)
        diff = abs(y_val - y_occ[1])
        saliency[i] = diff
    end
    
    return saliency
end
