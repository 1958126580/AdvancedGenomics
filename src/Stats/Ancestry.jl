"""
    Ancestry.jl

Ancestry-based Analysis tools.
IBD Segments, IBD-GRM, ROH.
Optimized for performance and thread safety.
"""

using Statistics
using LinearAlgebra
using DataFrames

"""
    detect_ibd_segments(H1, H2; min_len=50)

Detects IBD segments between all pairs of individuals using phased haplotypes.
H1, H2: n x m Haplotype matrices (0/1).
min_len: Minimum SNP length to consider as IBD.

Returns a DataFrame of segments: ID1, ID2, Start, End, Length.
"""
function detect_ibd_segments(H1::Matrix{Int}, H2::Matrix{Int}; min_len::Int=50)
    n, m = size(H1)
    
    # Thread-safe storage
    results = Vector{Tuple{Int, Int, Int, Int, Int}}()
    lck = ReentrantLock()
    
    Threads.@threads :dynamic for i in 1:n
        local_res = Vector{Tuple{Int, Int, Int, Int, Int}}()
        
        for j in i+1:n
            pairs = [
                (view(H1, i, :), view(H1, j, :)),
                (view(H1, i, :), view(H2, j, :)),
                (view(H2, i, :), view(H1, j, :)),
                (view(H2, i, :), view(H2, j, :))
            ]
            
            for (h_a, h_b) in pairs
                start_pos = -1
                current_len = 0
                
                for k in 1:m
                    if h_a[k] == h_b[k]
                        if current_len == 0
                            start_pos = k
                        end
                        current_len += 1
                    else
                        if current_len >= min_len
                            push!(local_res, (i, j, start_pos, k-1, current_len))
                        end
                        current_len = 0
                    end
                end
                if current_len >= min_len
                    push!(local_res, (i, j, start_pos, m, current_len))
                end
            end
        end
        
        # Merge to main results with lock
        lock(lck) do
            append!(results, local_res)
        end
    end
    
    id1 = [r[1] for r in results]
    id2 = [r[2] for r in results]
    start_p = [r[3] for r in results]
    end_p = [r[4] for r in results]
    len_p = [r[5] for r in results]
    
    segments = DataFrame(ID1=id1, ID2=id2, Start=start_p, End=end_p, Length=len_p)
    return segments
end

"""
    build_ibd_grm(segments, n, m)

Builds IBD-based GRM.
K_ij = Sum(Length_IBD) / (2 * Genome_Length)
"""
function build_ibd_grm(segments::DataFrame, n::Int, m::Int)
    K = zeros(Float64, n, n)
    
    # Diagonal = 1.0
    for i in 1:n
        K[i, i] = 1.0
    end
    
    # Fill off-diagonal
    # Group by ID1, ID2
    # We can iterate the DataFrame
    
    # Optimization: Use a dictionary or direct indexing if n is small.
    # Since n might be large, direct indexing into K is fine (O(N_segs)).
    
    for row in eachrow(segments)
        i = row.ID1
        j = row.ID2
        len = row.Length
        
        # Add to K
        # Normalized by 2 * m (2 haplotypes per individual, total length m)
        # Note: If we compare 4 pairs, the total space is 4 * m.
        # But we want "Proportion of genome shared IBD".
        # If I share 1 haplotype fully with my father, I share 50% of my genome.
        # My total IBD length would be m (from one pair) + 0 (others).
        # So sum(len) = m.
        # K should be 0.5.
        # So K = sum(len) / (2 * m).
        
        val = len / (2.0 * m)
        K[i, j] += val
        K[j, i] += val
    end
    
    return K
end

"""
    calculate_froh(G; min_len=50)

Calculates Inbreeding Coefficient based on Runs of Homozygosity (F_ROH).
G: Genotype Matrix (0, 1, 2)
"""
function calculate_froh(G::GenotypeMatrix; min_len::Int=50)
    M = G.data
    n, m = size(M)
    
    froh = zeros(Float64, n)
    
    Threads.@threads for i in 1:n
        total_roh_len = 0
        current_len = 0
        
        for j in 1:m
            g = M[i, j]
            # Check for homozygosity (0 or 2)
            # Treat missing as break? Or ignore?
            # Standard: Treat as break or impute. Here break.
            
            if !isnan(g) && (g == 0.0 || g == 2.0)
                current_len += 1
            else
                if current_len >= min_len
                    total_roh_len += current_len
                end
                current_len = 0
            end
        end
        if current_len >= min_len
            total_roh_len += current_len
        end
        
        froh[i] = total_roh_len / m
    end
    
    return froh
end
