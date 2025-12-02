"""
    SingleStep.jl

Utilities for Single-Step GBLUP (SS-GBLUP).
Constructs the H-inverse matrix combining Pedigree (A) and Genomic (G) relationships.
"""

using LinearAlgebra
using SparseArrays

"""
    build_A_inverse(pedigree::DataFrame)

Constructs the inverse of the numerator relationship matrix (A^-1) from pedigree.
Simple implementation for demonstration (Henderson's rules).
"""
function build_A_inverse(pedigree::DataFrame)
    # Assumes columns: ID, Sire, Dam
    # IDs must be 1..n renumbered
    n = nrow(pedigree)
    A_inv = spzeros(n, n)
    
    # Henderson's rules
    # dii = 2 / (1 - 0.25(known_parents)) ? No, simpler:
    # b = 4 / (2 + num_parents_unknown) ?
    # Standard:
    # Both unknown: d=2, b=1
    # One known: d=4/3, b=2/3
    # Both known: d=1, b=0.5
    
    for i in 1:n
        s = pedigree.Sire[i]
        d = pedigree.Dam[i]
        
        s_known = !ismissing(s) && s > 0
        d_known = !ismissing(d) && d > 0
        
        if !s_known && !d_known
            alpha = 1.0/2.0 # Variance of Mendelian sampling? 
            # Diagonal element += 2
            # Actually, simpler:
            # v = 1 - 0.25(Fs + Fd). Assume F=0 for base.
            v = 1.0
            A_inv[i, i] += 1.0/v
        elseif s_known && !d_known
            v = 0.75
            val = 1.0/v
            A_inv[i, i] += val
            A_inv[s, s] += 0.25 * val
            A_inv[i, s] -= 0.5 * val
            A_inv[s, i] -= 0.5 * val
        elseif !s_known && d_known
            v = 0.75
            val = 1.0/v
            A_inv[i, i] += val
            A_inv[d, d] += 0.25 * val
            A_inv[i, d] -= 0.5 * val
            A_inv[d, i] -= 0.5 * val
        else
            v = 0.5
            val = 1.0/v
            A_inv[i, i] += val
            A_inv[s, s] += 0.25 * val
            A_inv[d, d] += 0.25 * val
            A_inv[i, s] -= 0.5 * val
            A_inv[s, i] -= 0.5 * val
            A_inv[i, d] -= 0.5 * val
            A_inv[d, i] -= 0.5 * val
            A_inv[s, d] += 0.25 * val
            A_inv[d, s] += 0.25 * val
        end
    end
    
    return A_inv
end

"""
    build_H_inverse(A_inv, G_inv, genotyped_indices; w=0.05)

Constructs H^-1 for SS-GBLUP.
H^-1 = A^-1 + [0 0; 0 G^-1 - A_22^-1]
w: Weight for polygenic effect (blending G with A)
"""
function build_H_inverse(A_inv::AbstractMatrix, G::AbstractMatrix, genotyped_indices::Vector{Int}; w=0.05)
    # G_blend = (1-w)G + wA_22
    # For simplicity, we assume G is already inverted or we invert it here.
    # A_22 is the block of A for genotyped animals.
    # Inverting A_22 explicitly is expensive.
    # Standard single-step uses: H^-1 = A^-1 + [ 0 0 ; 0 (G^-1 - A_22^-1) ]
    
    G_inv = inv(G) # Expensive O(n^3)
    
    # Extract A_22 (relationship matrix among genotyped animals) and compute its inverse
    # This implements the standard single-step GBLUP H^-1 formula:
    # H^-1 = A^-1 + [ 0 0 ; 0 (G^-1 - A_22^-1) ]
    
    H_inv = Matrix(A_inv) # Convert to dense for addition
    
    # Compute A_22 from the full A matrix
    A = inv(Matrix(A_inv))
    A_22 = A[genotyped_indices, genotyped_indices]
    A_22_inv = inv(A_22)
    
    diff = G_inv - A_22_inv
    
    for (i, idx_i) in enumerate(genotyped_indices)
        for (j, idx_j) in enumerate(genotyped_indices)
            H_inv[idx_i, idx_j] += diff[i, j]
        end
    end
    
    return H_inv
end
