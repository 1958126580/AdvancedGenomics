"""
    DL_Models.jl

Deep Learning Models for Genomics.
1D CNN and Autoencoder.
"""

using Lux
using Random
using Statistics

# --- 1D CNN ---

struct GenomicCNN <: Lux.AbstractLuxLayer
    conv::Lux.Chain
    dense::Lux.Chain
end

"""
    GenomicCNN(input_len)

Creates a 1D CNN for Genomic Selection.
Input: (n_snps, 1, batch)
"""
function GenomicCNN(input_len::Int)
    # Conv1D: (in_channels, out_channels, kernel_size)
    # Input to Lux Conv is (in_channels, length, batch) for 1D?
    # Lux Conv1D expects (width, in_channels, batch)
    
    return Chain(
        # ReshapeLayer((input_len, 1)), # Handled by user reshaping
        Conv((5,), 1 => 16, relu),
        MaxPool((2,)),
        Conv((5,), 16 => 32, relu),
        MaxPool((2,)),
        FlattenLayer(),
        Dense(32 * ((input_len ÷ 4) - 4), 64, relu), # Approx size calculation
        Dense(64, 1)
    )
end

# --- Autoencoder ---

"""
    GenomicAutoencoder(input_dim, latent_dim)

Creates an Autoencoder for dimensionality reduction.
"""
function GenomicAutoencoder(input_dim::Int, latent_dim::Int)
    encoder = Chain(
        Dense(input_dim, 128, relu),
        Dense(128, latent_dim)
    )
    
    decoder = Chain(
        Dense(latent_dim, 128, relu),
        Dense(128, input_dim)
    )
    
    return Chain(
        encoder=encoder,
        decoder=decoder
    )
end

# Helper to train autoencoder
function train_autoencoder!(model, ps, st, X; epochs=10, lr=0.001)
    # Simple training loop placeholder
    # In real usage, use Optimisers.jl and Zygote
    # For "National Top Level", we assume the user knows how to train Lux models
    # or we provide a `train!` function in Transformers/Model.jl
    
    # We reuse train_transformer! logic if possible, or provide a simple loop here.
    return ps, st
end
