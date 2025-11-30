module AdvancedGenomics

using LinearAlgebra
using Statistics
using Random
using DataFrames
using CSV
using Lux
# using CUDA # Removed as it is a weak dependency

# Export Core Types
export GenotypeMatrix, PhenotypeData, MultiOmicsData, Pedigree

# Export IO Functions
export read_genotypes, read_phenotypes, read_omics, read_pedigree

# Export Statistical Functions
export build_grm, build_interaction_kernel, run_gwas, run_lmm
export build_rbf_kernel, build_poly_kernel, build_ibs_kernel
export estimate_vc_reml
export build_A, build_A_inverse
export simulate_genotypes, simulate_phenotypes, simulate_omics, simulate_multi_omics
export gaussian_copula_fit, pobs
export run_sire_model, run_test_day_model
export selection_index, predict_genetic_gain, predict_response_to_selection
export read_gff, annotate_snps
export hypergeometric_test, run_pathway_enrichment
export run_pca, run_farmcpu
export meta_analysis_ivw
export simple_fine_mapping
export build_dominance_kernel
export phase_genotypes, build_haplotype_matrix, build_haplotype_kernel
export read_plink, write_bed
export run_logistic_gwas, clump_snps, estimate_ibd_plink
export detect_ibd_segments, build_ibd_grm, calculate_froh
export random_forest, predict_rf, gradient_boosting, predict_gbm
export GenomicCNN, GenomicAutoencoder
export interval_mapping, permutation_test_qtl
export calculate_bayes_factor, wavelet_denoising, conformal_prediction
export simulated_annealing, genetic_algorithm_select, ant_colony_select
export optimal_contribution_selection
export run_gwas_pipeline, run_gs_pipeline
export cross_validation
export geweke_diagnostic, effective_sample_size, autocorrelation
export geweke_diagnostic, effective_sample_size, autocorrelation
export lasso_cd, elastic_net_cd
export adjust_pvalues, filter_maf, hwe_test

# Export Models
export run_bayesA, run_bayesB, run_bayesC, run_bayesR, run_bayesian_lasso
export run_multitrait_lmm
export run_nnmm
export run_vi_bayesc
export run_threshold_model
export run_random_regression
export mr_ivw, mr_egger
export run_burden_test
export run_inla_lmm

# Export Deep Learning Models
export GenomicTransformer, GenomicGNN, train_transformer!
export saliency_map

# Export Visualization
export manhattan_plot, qq_plot, manhattan_plot_interactive, qq_plot_interactive, generate_html_report, build_MME, solve_MME, calculate_r2, ld_pruning, benjamini_hochberg, build_dominance_kernel, build_haplotype_kernel, build_grm, build_rbf_kernel, estimate_vc_reml

# Export HPC
export build_grm_gpu

# Include sub-modules
include("Core/Types.jl")
include("Core/PedModule.jl")
include("IO/Readers.jl")
include("IO/PlinkIO.jl")
include("Stats/Kernels.jl")
include("Stats/VarianceComponents.jl")
include("Stats/MME.jl")
include("Stats/Diagnostics.jl")
include("Stats/LD.jl")
include("Stats/MultipleTesting.jl")
include("Stats/CompressedSensing.jl")
include("Stats/Simulation.jl")
include("Stats/Copula.jl")
include("Bioinformatics/Annotation.jl")
include("Bioinformatics/Enrichment.jl")
include("Stats/GWAS.jl")
include("Stats/MetaAnalysis.jl")
include("Stats/FineMapping.jl")
include("Stats/Haplotypes.jl")
include("Stats/PlinkStats.jl")
include("Stats/Ancestry.jl")
include("Stats/ML.jl")
include("Stats/QTL.jl")
include("Stats/ModernStats.jl")
include("Stats/Optimization.jl")
include("Stats/SingleStep.jl")
include("Stats/QC.jl")
include("Stats/Validation.jl")
include("Pipeline/Workflows.jl")

include("Models/LMM.jl")
include("Models/RareVariants.jl")
include("Models/Bayesian.jl")
include("Models/Multivariate.jl")
include("Models/NNMM.jl")
include("Models/Variational.jl")
include("Models/Causal.jl")
include("Models/NonGaussian.jl")
include("Models/RepeatedMeasures.jl")
include("Models/INLA.jl")
include("Models/BreedingModels.jl")
include("Breeding/Scheme.jl")
include("Breeding/OperationsResearch.jl")
include("Transformers/Model.jl")
include("Transformers/DL_Models.jl")
include("Transformers/Graph.jl")
include("Transformers/XAI.jl")
include("HPC/Parallel.jl")
include("Vis/Plots.jl")
include("Vis/Reports.jl")
include("CLI/Interface.jl")
end # module AdvancedGenomics
