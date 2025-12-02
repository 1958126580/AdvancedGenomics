"""
    Model.jl

Transformer architecture for Genomic Prediction using Lux.jl.
Treats the genome (or a region) as a sequence of tokens (SNPs).
"""

using Lux
using Random
using NNlib

"""
    GenomicEmbedding(in_dims, out_dims)

Embeds SNP genotypes (0, 1, 2) into a vector space.
Input: Integer matrix of shape (seq_len, batch_size)
Output: Array of shape (out_dims, seq_len, batch_size)
"""
struct GenomicEmbedding{E} <: Lux.AbstractLuxLayer
    embedding::E
end

struct DropScores <: Lux.AbstractLuxLayer end
(::DropScores)(x, ps, st) = (x[1], st)

function GenomicEmbedding(in_dims::Int, out_dims::Int)
    return GenomicEmbedding(Embedding(in_dims => out_dims))
end

Lux.initialparameters(rng::AbstractRNG, l::GenomicEmbedding) = Lux.initialparameters(rng, l.embedding)
Lux.initialstates(rng::AbstractRNG, l::GenomicEmbedding) = Lux.initialstates(rng, l.embedding)

function (l::GenomicEmbedding)(x, ps, st)
    # x is (seq_len, batch_size)
    # Embedding expects indices.
    # We assume x contains 0, 1, 2. We shift to 1, 2, 3 for 1-based indexing if needed,
    # or just ensure input is 1, 2, 3.
    # Let's assume input is 1-based indices for genotypes: 1->0, 2->1, 3->2.
    return l.embedding(x, ps, st)
end

"""
    GenomicTransformer(vocab_size, embed_dim, num_heads, hidden_dim, num_layers)

Creates a Transformer model for genomic data.
"""
function GenomicTransformer(; vocab_size=3, embed_dim=64, num_heads=4, hidden_dim=128, num_layers=2)
    
    # Encoder Layer
    encoder_layer = Chain(
        MultiHeadAttention(embed_dim, nheads=num_heads),
        DropScores(),
        # LayerNorm((1,)),
        Dense(embed_dim, hidden_dim, relu),
        Dense(hidden_dim, embed_dim)
        # LayerNorm((1,))
    )
    
    # Stack layers? Lux doesn't have a direct TransformerBlock stack utility easily accessible 
    # without defining the residual connections explicitly.
    # For brevity, we define a simple structure:
    # Embedding -> [Attention -> FeedForward] x N -> GlobalPool -> Dense -> Output
    
    layers = Vector{Any}()
    push!(layers, GenomicEmbedding(vocab_size, embed_dim))
    
    for _ in 1:num_layers
        # Simplified block: Attention -> Norm -> Dense -> Norm
        # Note: A real transformer has residuals. Lux Chain doesn't do residuals automatically.
        # We would need a custom SkipConnection layer.
        push!(layers, SkipConnection(
            Chain(MultiHeadAttention(embed_dim, nheads=num_heads), DropScores()), # LayerNorm((1,))), 
            +
        ))
        push!(layers, SkipConnection(
            Chain(Dense(embed_dim, hidden_dim, relu), Dense(hidden_dim, embed_dim)), # LayerNorm((1,))),
            +
        ))
    end
    
    # Prediction head
    push!(layers, WrappedFunction(x -> mean(x, dims=2))) # (embed_dim, 1, batch)
    push!(layers, FlattenLayer())
    push!(layers, Dense(embed_dim, 1)) # Regression output
    
    return Chain(layers...)
end

"""
    train_transformer!(model, X_train, y_train; epochs=10, lr=0.001, batch_size=32)

Complete training loop for Genomic Transformer.
Uses Adam optimizer and MSE loss for regression tasks.
"""
function train_transformer!(model, X_train::Matrix, y_train::Vector{Float64}; 
                           epochs::Int=10, 
                           lr::Float64=0.001, 
                           batch_size::Int=32)
    # Initialize parameters and state
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)
    
    # Setup optimizer
    opt_state = Optimisers.setup(Optimisers.Adam(lr), ps)
    
    n_samples = size(X_train, 2)
    n_batches = ceil(Int, n_samples / batch_size)
    
    # Training loop
    for epoch in 1:epochs
        total_loss = 0.0
        
        # Shuffle data
        perm = randperm(n_samples)
        X_shuffled = X_train[:, perm]
        y_shuffled = y_train[perm]
        
        for batch_idx in 1:n_batches
            # Get batch
            start_idx = (batch_idx - 1) * batch_size + 1
            end_idx = min(batch_idx * batch_size, n_samples)
            X_batch = X_shuffled[:, start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            # Compute loss and gradients
            loss, grads = Zygote.withgradient(ps) do p
                # Forward pass
                y_pred, _ = model(X_batch, p, st)
                
                # MSE loss
                mse = mean((vec(y_pred) .- y_batch).^2)
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
    
    return ps, st
end
