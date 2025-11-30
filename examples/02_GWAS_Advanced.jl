# ==============================================================================
# Example 02: Advanced GWAS & Fine-Mapping
# ==============================================================================
# This script demonstrates advanced GWAS techniques including logistic regression
# for binary traits, burden tests for rare variants, fine-mapping, and meta-analysis.
# ==============================================================================

using AdvancedGenomics
using Random
using DataFrames
using Statistics

Random.seed!(123)

println("--- Starting Example 02: Advanced GWAS & Fine-Mapping ---")

# ------------------------------------------------------------------------------
# Step 1: Logistic GWAS for Binary Traits
# ------------------------------------------------------------------------------
println("\n[Step 1] Logistic GWAS (Case-Control Study)...")

# Simulate genotypes and a binary phenotype (0 = Control, 1 = Case)
n_ind = 500
n_snps = 5000
genotypes = simulate_genotypes(n_ind, n_snps)
# Simulate binary phenotype: thresholding a continuous liability
liability = simulate_phenotypes(genotypes, 0.4).y
# ------------------------------------------------------------------------------
println("\n[Step 2] Burden Test for Rare Variants...")

# Simulate rare variants (MAF < 0.01)
genotypes_rare = simulate_genotypes(n_ind, 1000)
# Artificially lower MAF for simulation
genotypes_rare[genotypes_rare .> 0] .= 0
for i in 1:1000
    idx = rand(1:n_ind, 5) # Only 5 carriers per SNP
    genotypes_rare[idx, i] .= 1.0
end

# Define a "gene" region (e.g., SNPs 1-50 belong to Gene A)
gene_regions = [1:50, 51:100, 101:150]
burden_p_values = Float64[]
# ------------------------------------------------------------------------------
# Step 3: Meta-Analysis (IVW)
# ------------------------------------------------------------------------------
println("\n[Step 3] Meta-Analysis (Inverse Variance Weighted)...")

# Simulate summary statistics from two independent studies
n_snps_meta = 100
betas_study1 = randn(n_snps_meta)
se_study1 = rand(n_snps_meta) .* 0.1
betas_study2 = betas_study1 .+ randn(n_snps_meta) .* 0.05 # Correlated effects
se_study2 = rand(n_snps_meta) .* 0.12

# Perform Inverse Variance Weighted (IVW) Meta-Analysis
# meta_analysis_ivw expects vectors of effects for a single SNP across studies
# We iterate over all SNPs
meta_betas = Float64[]
for i in 1:n_snps_meta
    res = meta_analysis_ivw([betas_study1[i], betas_study2[i]], [se_study1[i], se_study2[i]])
    push!(meta_betas, res.beta)
end

println("  - Meta-Analysis completed.")
println("  - Mean Meta-Beta: $(mean(meta_betas))")

# ------------------------------------------------------------------------------
# Step 4: Fine-Mapping
# ------------------------------------------------------------------------------
println("\n[Step 4] Fine-Mapping Significant Loci...")

# Assume we found a significant locus with 50 SNPs in high LD
# We want to identify the causal SNP(s) within this locus
locus_snps = simulate_genotypes(n_ind, 50)
locus_pheno = simulate_phenotypes(locus_snps, 0.1, n_qtl=1).y # 1 causal SNP

# Run Simple Fine-Mapping (e.g., based on posterior probabilities or P-values)
# simple_fine_mapping(z_scores, ld_matrix)
# Calculate Z-scores (approximate from correlation)
correlations = [cor(locus_snps[:, i], locus_pheno) for i in 1:size(locus_snps, 2)]
z_scores = correlations .* sqrt(n_ind)
ld_matrix = cor(locus_snps)

pip = simple_fine_mapping(z_scores, ld_matrix)
top_snp_index = argmax(pip)
posterior_prob = pip[top_snp_index]

println("  - Fine-Mapping completed.")
println("  - Top Candidate SNP Index: $top_snp_index")
println("  - Posterior Probability: $(round(posterior_prob, digits=4))")

println("\n--- Example 02 Completed Successfully ---")
