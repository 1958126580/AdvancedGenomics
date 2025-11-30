"""
    LD.jl

Linkage Disequilibrium (LD) module.
Includes r^2 calculation and LD pruning.
"""

using Statistics
using LinearAlgebra

"""
    calculate_r2(g1, g2)

Computes pairwise LD (r^2) between two SNP vectors.
g1, g2: Genotype vectors (0, 1, 2).
"""
function calculate_r2(g1::AbstractVector, g2::AbstractVector)
    # r^2 is simply the squared correlation coefficient
    c = cor(g1, g2)
    return c^2
end

"""
    ld_pruning(G, threshold; window_size=50)

Performs LD pruning (clumping) on the genotype matrix.
Greedy algorithm:
1. Start with first SNP.
2. Calculate r^2 with subsequent SNPs in window.
3. Remove SNPs with r^2 > threshold.
4. Move to next kept SNP.

Returns: (G_pruned, kept_indices)
"""
function ld_pruning(G::AbstractMatrix, threshold::Float64; window_size::Int=50)
    n, m = size(G)
    kept_indices = Int[]
    removed = Set{Int}()
    
    # Sort by position? Assuming G is already sorted by position.
    
    for i in 1:m
        if i in removed
            continue
        end
        
        push!(kept_indices, i)
        
        # Check window
        end_idx = min(m, i + window_size)
        
        for j in (i+1):end_idx
            if j in removed
                continue
            end
            
            r2 = calculate_r2(G[:, i], G[:, j])
            
            if r2 > threshold
                push!(removed, j)
            end
        end
    end
    
    return (G[:, kept_indices], kept_indices)
end
