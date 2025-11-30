# ==============================================================================
# Example 10: HPC & Population Genetics
# ==============================================================================
# This script demonstrates High-Performance Computing (GPU/Parallel) features
# and Population Genetics analyses (Phasing, IBD, Ancestry).
# ==============================================================================

using AdvancedGenomics
using Random
using Statistics
using DataFrames

Random.seed!(606)

println("--- Starting Example 10: HPC & Population Genetics ---")

# ------------------------------------------------------------------------------
# Step 1: Haplotype Phasing
# ------------------------------------------------------------------------------
println("\n[Step 1] Haplotype Phasing...")

n_ind = 100
n_snps = 100
genotypes = simulate_genotypes(n_ind, n_snps)

# Wrap in GenotypeMatrix
ids = ["Ind_$i" for i in 1:n_ind]
snps_ids = ["SNP_$i" for i in 1:n_snps]
G_obj = GenotypeMatrix(genotypes, ids, snps_ids)

# Phase genotypes into haplotypes (infer paternal/maternal origin)
# Use smaller parameters for demonstration speed
haplotypes = phase_genotypes(G_obj, n_states=5, n_iter=2)

println("  - Phasing completed.")
println("  - Haplotype Matrix Size: $(size(haplotypes.H1)) (2x individuals)")

# ------------------------------------------------------------------------------
# Step 2: IBD Detection & Ancestry
# ------------------------------------------------------------------------------
println("\n[Step 2] IBD Detection & Ancestry Inference...")

# Detect Identity-by-Descent (IBD) segments between individuals
# IBD segments indicate shared ancestry
ibd_segments = detect_ibd_segments(haplotypes.H1, haplotypes.H2)

println("  - Detected $(nrow(ibd_segments)) IBD segments.")

# Build IBD-based Genomic Relationship Matrix (GRM)
K_ibd = build_ibd_grm(ibd_segments, n_ind, n_snps)
println("  - IBD-GRM constructed.")

# Calculate Inbreeding Coefficients based on Runs of Homozygosity (F_ROH)
f_roh = calculate_froh(G_obj)
println("  - Mean F_ROH: $(round(mean(f_roh), digits=4))")

# ------------------------------------------------------------------------------
# Step 3: GPU Acceleration (Demo)
# ------------------------------------------------------------------------------
println("\n[Step 3] GPU Acceleration Demo...")

# Check if CUDA is available (simulated check for this example)
cuda_available = false
try
    using CUDA
    if CUDA.functional()
        cuda_available = true
    end
catch
    println("  - CUDA not loaded or available. Skipping actual GPU execution.")
end

if cuda_available
    println("  - GPU Detected! Running Benchmark...")
    
    # Benchmark CPU vs GPU GRM construction
    n_large = 2000
    m_large = 10000
    G_large = simulate_genotypes(n_large, m_large)
    
    println("    -> Running CPU GRM...")
    time_cpu = @elapsed build_grm(G_large)
    println("       CPU Time: $(round(time_cpu, digits=2))s")
    
    println("    -> Running GPU GRM...")
    time_gpu = @elapsed build_grm_gpu(G_large)
    println("       GPU Time: $(round(time_gpu, digits=2))s")
    
    println("    -> Speedup: $(round(time_cpu / time_gpu, digits=1))x")
    
    # Run GPU GWAS
    run_gwas_gpu(G_large, randn(n_large))
    println("    -> GPU GWAS completed.")
else
    println("  - GPU functions `build_grm_gpu` and `run_gwas_gpu` are available")
    println("    when `using CUDA` is present and a GPU is detected.")
    println("  - Please run this script on a GPU-enabled machine to see the speedup.")
end

println("\n--- Example 10 Completed Successfully ---")
