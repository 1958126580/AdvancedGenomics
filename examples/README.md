# AdvancedGenomics.jl Examples

This directory contains a comprehensive suite of "Gold Standard" examples demonstrating the full capabilities of `AdvancedGenomics.jl`.

## Getting Started

To run any example, start Julia in the project root and include the script:

```bash
julia --project=. examples/01_GWAS_Standard.jl
```

## List of Examples

| Script                   | Description            | Key Features                                                                        |
| :----------------------- | :--------------------- | :---------------------------------------------------------------------------------- |
| **01_GWAS_Standard.jl**  | Standard GWAS Pipeline | `run_farmcpu`, `filter_maf`, `run_pca`, `manhattan_plot_interactive`                |
| **02_GWAS_Advanced.jl**  | Advanced GWAS          | `run_logistic_gwas`, `run_burden_test`, `simple_fine_mapping`, `meta_analysis_ivw`  |
| **03_GS_Kernels_ML.jl**  | Genomic Selection (ML) | `build_rbf_kernel`, `random_forest`, `gradient_boosting`                            |
| **04_GS_Bayesian.jl**    | Bayesian Selection     | `run_bayesA`, `run_bayesC`, `run_vi_bayesc`, `geweke_diagnostic`                    |
| **05_Deep_Learning.jl**  | Deep Learning          | `GenomicTransformer`, `GenomicGNN`, `saliency_map`                                  |
| **06_Complex_Models.jl** | Complex Models         | `run_multitrait_lmm`, `run_threshold_model`, `run_random_regression`                |
| **07_Breeding.jl**       | Breeding Optimization  | `build_A`, `selection_index`, `optimal_contribution_selection`                      |
| **08_Multi_Omics.jl**    | Multi-Omics & Causal   | `gaussian_copula_fit`, `mr_ivw`, `run_pathway_enrichment`                           |
| **09_Modern_Stats.jl**   | Modern Statistics      | `lasso_cd`, `conformal_prediction`, `wavelet_denoising`, `genetic_algorithm_select` |
| **10_HPC_PopGen.jl**     | HPC & PopGen           | `build_grm_gpu`, `phase_genotypes`, `detect_ibd_segments`                           |

## Verification

You can run all examples sequentially to verify the installation:

```bash
julia --project=. examples/run_all_examples.jl
```
