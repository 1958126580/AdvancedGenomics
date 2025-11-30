"""
    Graph.jl

Graph Neural Networks (GNN) for Genomic Pathway Analysis.
Uses adjacency matrix (pathways) to structure the neural network.
"""

using Lux
using LinearAlgebra
using Statistics
using Random

"""
    GraphConv(in_dims, out_dims, adj)

Graph Convolutional Layer.
H' = ReLU(D^-0.5 A D^-0.5 H W)
"""
struct GraphConv{A, L} <: Lux.AbstractLuxLayer
    adj::A # Adjacency matrix (normalized)
    dense::L # Dense layer for W
end

function GraphConv(in_dims::Int, out_dims::Int, adj::AbstractMatrix)
    # Normalize Adjacency: D^-0.5 (A + I) D^-0.5
    A_hat = adj + I
    deg = vec(sum(A_hat, dims=1))
    D_inv_sqrt = Diagonal(1.0 ./ sqrt.(deg))
    norm_adj = D_inv_sqrt * A_hat * D_inv_sqrt
    
    return GraphConv(norm_adj, Dense(in_dims, out_dims))
end

Lux.initialparameters(rng::AbstractRNG, l::GraphConv) = Lux.initialparameters(rng, l.dense)
Lux.initialstates(rng::AbstractRNG, l::GraphConv) = Lux.initialstates(rng, l.dense)

function (l::GraphConv)(x, ps, st)
    # x: (features, nodes, batch) ?
    # Standard GCN: H is (nodes, features)
    # Lux Dense expects (features, batch).
    
    # Let's assume input x is (nodes, features)
    # We need to transpose for Dense?
    # Dense: W * x
    
    # AHW:
    # 1. HW: Apply Dense to each node features
    # 2. A(HW): Propagate
    
    # x is (in_dims, nodes) for one sample?
    # Let's assume x is (in_dims, nodes, batch)
    
    # Apply Dense to each node
    # Reshape to (in_dims, nodes * batch)
    s = size(x)
    x_flat = reshape(x, s[1], :)
    
    h_flat, st_dense = l.dense(x_flat, ps, st)
    h = reshape(h_flat, size(h_flat, 1), s[2], s[3]) # (out_dims, nodes, batch)
    
    # Propagate: A * H' (where H' is nodes x out_dims)
    # Our h is out_dims x nodes. So we want h * A' ?
    # (out, nodes) * (nodes, nodes) = (out, nodes)
    
    # A is symmetric usually.
    # We broadcast over batch
    
    # h_out = h * l.adj'
    # But h is 3D.
    
    h_out = similar(h)
    for b in 1:s[3]
        h_out[:, :, b] = h[:, :, b] * l.adj'
    end
    
    return h_out, st
end

"""
    GenomicGNN(adj, in_dims, hidden_dims, out_dims)

Creates a GNN model.
"""
function GenomicGNN(adj::Matrix{Float32}, in_dims::Int, hidden_dims::Int, out_dims::Int)
    return Chain(
        GraphConv(in_dims, hidden_dims, adj),
        GraphConv(hidden_dims, out_dims, adj),
        FlattenLayer(),
        Dense(out_dims * size(adj, 1), 1) # Readout
    )
end
