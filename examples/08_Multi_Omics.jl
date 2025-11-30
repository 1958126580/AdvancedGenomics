# ==============================================================================
# Example 08: Multi-Omics & Causal Inference
# ==============================================================================
# This script demonstrates the integration of multiple omics layers (Genomics,
# Transcriptomics) and Causal Inference using Mendelian Randomization (MR).
# ==============================================================================

using AdvancedGenomics
using Random
using Statistics

Random.seed!(404)

println("--- Starting Example 08: Multi-Omics & Causal Inference ---")

# ------------------------------------------------------------------------------
# Step 1: Multi-Omics Simulation
# ------------------------------------------------------------------------------
println("\n[Step 1] Simulating Multi-Omics Data...")

n_ind = 200
n_snps = 500
n_genes = 50

# Simulate Genotypes (G), Transcriptome (T), and Phenotype (Y)
# G -> T -> Y (Causal chain)
# simulate_multi_omics(n, m_snps, m_omics)
multi_omics = simulate_multi_omics(n_ind, n_snps, n_genes)

genotypes = multi_omics.G
expression = multi_omics.M
phenotypes = multi_omics.y

println("  - Simulated Genotypes, Expression, and Phenotypes.")

# ------------------------------------------------------------------------------
println("\n[Step 3] Mendelian Randomization (Causal Inference)...")

# We want to test if Gene Expression (Exposure) causes the Phenotype (Outcome)
# using SNPs as Instrumental Variables (IVs).

# 1. GWAS for Exposure (G -> T)
# Wrap genotypes
ids = ["Ind_$i" for i in 1:n_ind]
snps_ids = ["SNP_$i" for i in 1:n_snps]
G_obj = GenotypeMatrix(genotypes, ids, snps_ids)

gwas_exp = run_farmcpu(expression[:, 1], G_obj, zeros(n_ind, 0))
sig_ivs = findall(gwas_exp.p_values .< 0.05)

if isempty(sig_ivs)
    println("  - No significant IVs found. Skipping MR.")
else
    # 2. GWAS for Outcome (G -> Y)
    gwas_out = run_farmcpu(phenotypes, G_obj, zeros(n_ind, 0))
    
    # Extract Beta coefficients for IVs
    beta_exp = gwas_exp.p_values[sig_ivs] # Placeholder
    se_exp = rand(length(sig_ivs)) .* 0.1 # Placeholder
    beta_out = gwas_out.p_values[sig_ivs] # Placeholder
    se_out = rand(length(sig_ivs)) .* 0.1
    
    # 3. Run MR-IVW (Inverse Variance Weighted)
    # Estimate causal effect of T on Y
    # mr_ivw(beta_exp, se_exp, beta_out, se_out)
    mr_result = mr_ivw(beta_exp, se_exp, beta_out, se_out)
    
    println("  - MR-IVW completed.")
    println("  - Causal Effect Estimate: $(round(mr_result.beta, digits=4))")
    println("  - P-value: $(mr_result.p_value)")
    
    # 4. Run MR-Egger (Checks for pleiotropy)
    mr_egger_res = mr_egger(beta_exp, se_exp, beta_out, se_out)
    println("  - MR-Egger Intercept (Pleiotropy): $(round(mr_egger_res.intercept, digits=4))")
end

# ------------------------------------------------------------------------------
# Step 4: Pathway Enrichment
# ------------------------------------------------------------------------------
println("\n[Step 4] Pathway Enrichment Analysis...")

# Assume we identified 5 significant genes
sig_genes = Set(["GeneA", "GeneB", "GeneC", "GeneD", "GeneE"])

# Create dummy Pathway DB
pathway_db = Dict(
    "Pathway_1" => Set(["GeneA", "GeneB", "GeneX", "GeneY"]),
    "Pathway_2" => Set(["GeneC", "GeneD", "GeneZ", "GeneW"]),
    "Pathway_3" => Set(["GeneE", "GeneF", "GeneG", "GeneH"])
)

# Background genes (all genes)
background_genes = Set(["GeneA", "GeneB", "GeneC", "GeneD", "GeneE", "GeneF", "GeneG", "GeneH", "GeneX", "GeneY", "GeneZ", "GeneW"])

# Run enrichment
enrichment = run_pathway_enrichment(sig_genes, pathway_db, background_genes)

println("  - Enrichment Analysis completed.")
if !isempty(enrichment)
    println("  - Top Pathway: $(enrichment.Pathway[1]) (P=$(round(enrichment.P_Value[1], digits=4)))")
else
    println("  - No significant pathways found.")
end

println("\n--- Example 08 Completed Successfully ---")
