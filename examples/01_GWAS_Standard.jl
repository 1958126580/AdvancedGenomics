# ==============================================================================
# Example 01: Standard GWAS Pipeline
# ==============================================================================
# This script demonstrates a complete Genome-Wide Association Study (GWAS) pipeline
# using AdvancedGenomics.jl. It covers data simulation, quality control,
# population structure analysis, association testing, and visualization.
# ==============================================================================

using AdvancedGenomics
using Random
using DataFrames
using Statistics
using PlotlyJS

# Set random seed for reproducibility
Random.seed!(42)

println("--- Starting Example 01: Standard GWAS Pipeline ---")

# ------------------------------------------------------------------------------
# Step 1: Data Simulation
# ------------------------------------------------------------------------------
println("\n[Step 1] Simulating Genotypes and Phenotypes...")

# Simulate 500 individuals and 10,000 SNPs
n_ind = 500
n_snps = 10000
genotypes = simulate_genotypes(n_ind, n_snps)

# Simulate a quantitative phenotype with 10 causal SNPs (h2 = 0.5)
# We assume the first 10 SNPs are causal for demonstration purposes
phenotypes = simulate_phenotypes(genotypes, 0.5, n_qtl=10).y

println("  - Genotypes: $(size(genotypes, 1)) individuals x $(size(genotypes, 2)) SNPs")
println("  - Phenotypes: $(length(phenotypes)) observations")

# ------------------------------------------------------------------------------
# Step 2: Quality Control (QC)
# ------------------------------------------------------------------------------
println("\n[Step 2] Performing Quality Control...")

# Filter SNPs with Minor Allele Frequency (MAF) < 0.05
# This removes rare variants that might cause spurious associations
genotypes_qc, kept_indices_maf = filter_maf(genotypes, 0.05)
println("  - SNPs remaining after MAF filtering: $(size(genotypes_qc, 2))")

# Filter SNPs deviating from Hardy-Weinberg Equilibrium (HWE) p < 1e-6
# This removes genotyping errors or markers under strong selection
# hwe_test returns p-values. I need to filter manually.
# Ensure genotypes_qc is Matrix{Float64}
p_hwe = hwe_test(Matrix{Float64}(genotypes_qc))
keep_hwe = p_hwe .>= 1e-6
genotypes_qc = genotypes_qc[:, keep_hwe]
println("  - SNPs remaining after HWE filtering: $(size(genotypes_qc, 2))")

# ------------------------------------------------------------------------------
# Step 3: Population Structure Analysis
# ------------------------------------------------------------------------------
println("\n[Step 3] Analyzing Population Structure (PCA)...")

# Perform Principal Component Analysis (PCA) to capture population structure
# We use the top 3 Principal Components (PCs) as covariates in the GWAS model
# Wrap in GenotypeMatrix
ids_qc = ["Ind_$i" for i in 1:n_ind]
snps_qc = ["SNP_$i" for i in 1:size(genotypes_qc, 2)]
G_qc_obj = GenotypeMatrix(genotypes_qc, ids_qc, snps_qc)

pca_result = run_pca(G_qc_obj, k=3)
covariates = pca_result.projections
variance_explained = pca_result.eigenvalues ./ sum(pca_result.eigenvalues)

println("  - PCA completed. Top 3 PCs extracted.")
println("  - Variance explained by top 3 PCs: $(round.(variance_explained[1:3] * 100, digits=2))%")

# ------------------------------------------------------------------------------
# Step 4: Association Testing (FarmCPU)
# ------------------------------------------------------------------------------
println("\n[Step 4] Running Association Testing (FarmCPU)...")

# Run FarmCPU (Fixed and Random Model Circulating Probability Unification)
# This method iteratively controls for false positives and false negatives
# We include the PCA covariates to correct for stratification
# run_farmcpu(y, G, X_cov)
gwas_results = run_farmcpu(phenotypes, G_qc_obj, covariates)

# Extract P-values and SNP positions (simulated positions)
p_values = gwas_results.p_values
chr = sort(rand(1:5, size(genotypes_qc, 2))) # Simulated chromosomes 1-5
pos = sort(rand(1:1000000, size(genotypes_qc, 2))) # Simulated positions

println("  - GWAS completed.")
println("  - Top significant SNP P-value: $(minimum(p_values))")

# ------------------------------------------------------------------------------
# Step 5: Visualization
# ------------------------------------------------------------------------------
println("\n[Step 5] Generating Interactive Visualizations...")

# Manhattan Plot
# Visualizes the -log10(P-values) across the genome
manhattan_plt = manhattan_plot_interactive(p_values, chr, pos, title="Example 01: Manhattan Plot")
PlotlyJS.savefig(manhattan_plt, "examples/01_manhattan_plot.html")
println("  - Saved 'examples/01_manhattan_plot.html'")

# QQ Plot
# Visualizes the observed vs expected P-values to check for inflation
qq_plt = qq_plot_interactive(p_values, title="Example 01: QQ Plot")
PlotlyJS.savefig(qq_plt, "examples/01_qq_plot.html")
println("  - Saved 'examples/01_qq_plot.html'")

println("\n--- Example 01 Completed Successfully ---")
