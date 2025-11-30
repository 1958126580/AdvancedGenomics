"""
    QTL.jl

Quantitative Trait Locus (QTL) Mapping methods.
Interval Mapping and LOD Scores.
"""

using Statistics
using Distributions
using LinearAlgebra

"""
    interval_mapping(y, G, map_pos; step=1.0)

Performs Interval Mapping (IM) scan.
y: Phenotype vector
G: Genotype Matrix (0, 1, 2)
map_pos: Vector of marker positions (e.g., in cM or bp).
step: Step size for scanning (in same units as map_pos).

Returns:
- positions: Scanned positions
- lod_scores: LOD score at each position
"""
function interval_mapping(y::Vector{Float64}, G::GenotypeMatrix, map_pos::Vector{Float64}; step::Float64=1.0)
    n, m = size(G.data)
    
    # Define scan positions
    # For simplicity, we scan between min and max of provided map
    # In reality, we scan per chromosome.
    # We assume G is one chromosome for this function or map_pos is continuous.
    
    start_pos = minimum(map_pos)
    end_pos = maximum(map_pos)
    scan_pos = collect(start_pos:step:end_pos)
    
    lod_scores = zeros(length(scan_pos))
    
    # Null model (no QTL)
    # y = mu + e
    mu_null = mean(y)
    rss_null = sum((y .- mu_null).^2)
    
    # For each position
    Threads.@threads for i in 1:length(scan_pos)
        pos = scan_pos[i]
        
        # Find flanking markers
        # idx such that map_pos[idx] <= pos < map_pos[idx+1]
        idx = findlast(map_pos .<= pos)
        
        if idx === nothing || idx == m
            # At ends or outside, use nearest marker
            marker_idx = (idx === nothing) ? 1 : m
            # Single marker regression
            g = G.data[:, marker_idx]
            # Simple regression y = mu + beta*g + e
            # Ignore missing for now
            X = hcat(ones(n), g)
            try
                beta = X \ y
                y_pred = X * beta
                rss_alt = sum((y .- y_pred).^2)
                
                # LOD = (n/2) * log10(RSS0 / RSS1)
                lod = (n / 2.0) * log10(rss_null / rss_alt)
                lod_scores[i] = lod
            catch
                lod_scores[i] = 0.0
            end
        else
            # Interval Mapping
            # Flanking markers
            m_left = idx
            m_right = idx + 1
            
            pos_left = map_pos[m_left]
            pos_right = map_pos[m_right]
            
            # Recombination fraction (Haldane mapping function)
            # r = 0.5 * (1 - exp(-2 * d)) for d in Morgans
            # distance d
            dist = (pos_right - pos_left) # assuming cM?
            # If bp, need conversion. Let's assume input is cM for standard IM.
            # If bp, dist is large, r ~ 0.5.
            # Let's assume map_pos is generic distance.
            
            # Simple regression on flanking markers (Haley-Knott Regression approximation)
            # Expected genotype at pos given flanking markers
            # E[Q | M_left, M_right]
            
            # Weights depend on distance
            d_left = pos - pos_left
            d_right = pos_right - pos
            d_total = pos_right - pos_left
            
            if d_total < 1e-6
                w_left = 1.0
                w_right = 0.0
            else
                w_left = 1.0 - (d_left / d_total)
                w_right = 1.0 - (d_right / d_total)
            end
            
            # Impute Q
            g_left = G.data[:, m_left]
            g_right = G.data[:, m_right]
            
            # Linear interpolation of genotype (approx for dense markers)
            q_hat = w_left .* g_left .+ w_right .* g_right
            
            # Regression
            X = hcat(ones(n), q_hat)
            try
                beta = X \ y
                y_pred = X * beta
                rss_alt = sum((y .- y_pred).^2)
                
                lod = (n / 2.0) * log10(rss_null / rss_alt)
                lod_scores[i] = lod
            catch
                lod_scores[i] = 0.0
            end
        end
    end
    
    return (positions=scan_pos, lod_scores=lod_scores)
end

"""
    permutation_test_qtl(y, G, map_pos; n_perm=100, step=1.0)

Runs permutation test to find LOD threshold.
"""
function permutation_test_qtl(y::Vector{Float64}, G::GenotypeMatrix, map_pos::Vector{Float64}; n_perm::Int=100, step::Float64=1.0)
    max_lods = zeros(n_perm)
    
    # Original order G, permute y
    y_perm = copy(y)
    
    for p in 1:n_perm
        shuffle!(y_perm)
        res = interval_mapping(y_perm, G, map_pos, step=step)
        max_lods[p] = maximum(res.lod_scores)
    end
    
    # 95th percentile
    threshold = quantile(max_lods, 0.95)
    return threshold
end
