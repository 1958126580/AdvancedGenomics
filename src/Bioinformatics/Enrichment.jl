"""
    Enrichment.jl

Pathway Enrichment Analysis.
"""

using Statistics
using Distributions

"""
    hypergeometric_test(k, M, n, N)

Calculates p-value for enrichment (Hypergeometric test / Fisher's Exact).
k: Number of significant hits in pathway
M: Total number of genes in pathway
n: Total number of significant hits (genome-wide)
N: Total number of genes (genome-wide)

P(X >= k)
"""
function hypergeometric_test(k::Int, M::Int, n::Int, N::Int)
    # Julia's Hypergeometric(s, f, n)
    # s = successes in population (M)
    # f = failures in population (N - M)
    # n = sample size (n)
    
    dist = Hypergeometric(M, N - M, n)
    # P(X >= k) = 1 - P(X <= k-1) = ccdf(dist, k-1)
    p_value = ccdf(dist, k - 1)
    return p_value
end

"""
    run_pathway_enrichment(significant_genes, pathway_db, background_genes)

Runs enrichment for multiple pathways.
significant_genes: Set of gene IDs
pathway_db: Dict(PathwayID => Set(GeneIDs))
background_genes: Set of all gene IDs tested
"""
function run_pathway_enrichment(significant_genes::Set{String}, pathway_db::Dict{String, Set{String}}, background_genes::Set{String})
    results = DataFrame(Pathway=[], P_Value=[], Overlap=[])
    
    N = length(background_genes)
    n = length(intersect(significant_genes, background_genes))
    
    for (pid, genes) in pathway_db
        # Filter genes to background
        pathway_genes = intersect(genes, background_genes)
        M = length(pathway_genes)
        
        if M == 0 continue end
        
        overlap = intersect(significant_genes, pathway_genes)
        k = length(overlap)
        
        if k > 0
            p = hypergeometric_test(k, M, n, N)
            push!(results, (pid, p, k))
        end
    end
    
    # Sort by P-value
    sort!(results, :P_Value)
    return results
end
