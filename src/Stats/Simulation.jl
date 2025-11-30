"""
    Simulation.jl

Data Simulation module.
Generates synthetic Genotypes, Phenotypes, and Multi-omics data.
"""

using Random
using Distributions
using LinearAlgebra

"""
    simulate_genotypes(n, m; maf_range=(0.05, 0.5))

Simulates a Genotype Matrix (n individuals, m SNPs).
Alleles are drawn from Binomial(2, p).
"""
function simulate_genotypes(n::Int, m::Int; maf_range::Tuple{Float64, Float64}=(0.05, 0.5))
    G = zeros(Float64, n, m)
    mafs = rand(Uniform(maf_range[1], maf_range[2]), m)
    
    for j in 1:m
        p = mafs[j]
        # 0, 1, 2
        G[:, j] = rand(Binomial(2, p), n)
    end
    
    return G
end

"""
    simulate_phenotypes(G, h2; n_qtl=10)

Simulates phenotypes with a specified heritability.
y = G_qtl * beta + e
"""
function simulate_phenotypes(G::Matrix{Float64}, h2::Float64; n_qtl::Int=10)
    n, m = size(G)
    
    # Select QTLs
    qtl_indices = sort(randperm(m)[1:n_qtl])
    G_qtl = G[:, qtl_indices]
    
    # Effects
    beta = randn(n_qtl)
    
    # Genetic Value
    g = G_qtl * beta
    var_g = var(g)
    
    if var_g == 0
        var_g = 1.0 # Edge case
    end
    
    # Residual Variance
    # h2 = var_g / (var_g + var_e)
    # var_e = var_g * (1 - h2) / h2
    var_e = var_g * (1.0 - h2) / h2
    
    e = randn(n) * sqrt(var_e)
    
    y = g + e
    
    return (y=y, qtl_indices=qtl_indices, beta=beta, h2_real=var(g)/var(y))
end

"""
    simulate_omics(G, h2_omics; n_eqtl=10)

Simulates Omics data (e.g., Gene Expression).
M = G_eqtl * alpha + delta
"""
function simulate_omics(G::Matrix{Float64}, h2_omics::Float64; n_eqtl::Int=10)
    # Similar to phenotype simulation but returns a matrix M (n x 1 for single feature, or multiple)
    # Let's simulate a single omics feature for now, or a matrix?
    # Usually we simulate a matrix of omics features.
    # Let's simulate 1 feature for simplicity in this function, or allow multiple.
    
    return simulate_phenotypes(G, h2_omics; n_qtl=n_eqtl)
end

"""
    simulate_multi_omics(n, m_snps, m_omics; h2_g=0.5, h2_m=0.3)

Simulates a full dataset: Genotypes, Omics, and Phenotypes.
Phenotype depends on Genotypes and Omics.
y = G*b + M*a + e
"""
function simulate_multi_omics(n::Int, m_snps::Int, m_omics::Int; h2_g::Float64=0.3, h2_m::Float64=0.2)
    # 1. Genotypes
    G = simulate_genotypes(n, m_snps)
    
    # 2. Omics (Dependent on G)
    # M = G * A + E_m
    # Let's simplify: Each omics trait is controlled by some SNPs
    M = zeros(Float64, n, m_omics)
    for i in 1:m_omics
        res = simulate_phenotypes(G, 0.5; n_qtl=5)
        M[:, i] = res.y
    end
    
    # 3. Phenotype (Dependent on G and M)
    # y = G_qtl * b + M_qtl * a + e
    
    # Direct genetic effects
    qtl_indices = sort(randperm(m_snps)[1:10])
    beta = randn(10)
    g_direct = G[:, qtl_indices] * beta
    
    # Omics effects
    omics_indices = sort(randperm(m_omics)[1:5])
    alpha = randn(5)
    g_omics = M[:, omics_indices] * alpha
    
    total_g = g_direct + g_omics
    var_g = var(total_g)
    
    # Total heritability (G+M)
    h2_total = h2_g + h2_m # Rough approximation
    if h2_total >= 1.0 h2_total = 0.9 end
    
    var_e = var_g * (1.0 - h2_total) / h2_total
    e = randn(n) * sqrt(var_e)
    
    y = total_g + e
    
    return (G=G, M=M, y=y, qtl=qtl_indices, omics_qtl=omics_indices)
end
