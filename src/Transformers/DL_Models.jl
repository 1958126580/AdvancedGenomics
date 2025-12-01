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
"""
    train_autoencoder!(model, X; epochs=10, lr=0.001, batch_size=32, use_gpu=false)

Trains an autoencoder model using Adam optimizer.

# Arguments
- `model`: Lux model (encoder-decoder chain)
- `X`: Training data (features × samples)
- `epochs`: Number of training epochs
- `lr`: Learning rate
- `batch_size`: Mini-batch size
- `use_gpu`: Whether to use GPU acceleration

# Returns
- Trained parameters and state
"""
function train_autoencoder!(model, X::Matrix{Float32}; 
                           epochs::Int=10, 
                           lr::Float64=0.001, 
                           batch_size::Int=32,
                           use_gpu::Bool=false)
    using Optimisers
    using Zygote
    
    # Initialize parameters and state
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)
    
    # Move to GPU if requested
    if use_gpu
        try
            using CUDA
            if CUDA.functional()
                ps = ps |> gpu
                st = st |> gpu
                X = X |> gpu
            else
                @warn "GPU requested but CUDA not functional. Using CPU."
                use_gpu = false
            end
        catch
            @warn "GPU requested but CUDA not available. Using CPU."
            use_gpu = false
        end
    end
    
    # Setup optimizer
    opt_state = Optimisers.setup(Optimisers.Adam(lr), ps)
    
    n_samples = size(X, 2)
    n_batches = ceil(Int, n_samples / batch_size)
    
    # Training loop
    for epoch in 1:epochs
        total_loss = 0.0
        
        # Shuffle data
        perm = randperm(n_samples)
        X_shuffled = X[:, perm]
        
        for batch_idx in 1:n_batches
            # Get batch
            start_idx = (batch_idx - 1) * batch_size + 1
            end_idx = min(batch_idx * batch_size, n_samples)
            X_batch = X_shuffled[:, start_idx:end_idx]
            
            # Compute loss and gradients
            loss, grads = Zygote.withgradient(ps) do p
                # Forward pass
                X_recon, st_new = model(X_batch, p, st)
                
                # MSE loss
                mse = mean((X_batch .- X_recon).^2)
                return mse
            end
            
            # Update parameters
            opt_state, ps = Optimisers.update(opt_state, ps, grads[1])
            
            total_loss += loss
        end
        
        avg_loss = total_loss / n_batches
        if epoch % 10 == 0 || epoch == 1
            println("Epoch $epoch/$epochs - Loss: $(round(avg_loss, digits=6))")
        end
    end
    
    # Move back to CPU if needed
    if use_gpu
        ps = ps |> cpu
        st = st |> cpu
    end
    
    return ps, st
end
