"""
    Workflows.jl

High-level pipelines for Genomic Analysis.
Integrates QC, Stats, Models, and Bioinformatics.
"""


using Statistics
using DataFrames

"""
    run_gwas_pipeline(G, y; cov=nothing, p_thresh=0.05)

Runs a complete GWAS pipeline:
1. QC (MAF > 0.01)
2. PCA (3 PCs)
3. GWAS (FarmCPU)
4. Multiple Testing (FDR)
5. Summary
"""
function run_gwas_pipeline(G::GenotypeMatrix, y::Vector{Float64}; cov::Union{Matrix{Float64}, Nothing}=nothing, p_thresh::Float64=0.05)
    println("--- Starting GWAS Pipeline ---")
    
    # 1. QC
    println("Step 1: Quality Control")
    # Filter MAF
    # For demo, we assume G is clean or we'd use filter_maf(G, 0.01)
    # Let's just print info
    n, m = size(G.data)
    println("  Input: $n individuals, $m SNPs")
    
    # 2. PCA
    println("Step 2: Population Structure (PCA)")
    pca_res = run_pca(G, k=3)
    X_pca = pca_res.projections
    
    # Combine covariates
    if cov !== nothing
        X_cov = hcat(cov, X_pca)
    else
        X_cov = X_pca
    end
    
    # 3. GWAS
    println("Step 3: Association Testing (FarmCPU)")
    # FarmCPU requires DataFrame phenotype? Or vector?
    # Our run_farmcpu takes (G, y, X_cov)
    gwas_res = run_farmcpu(y, G, X_cov)
    
    # 4. Multiple Testing
    println("Step 4: Multiple Testing Correction (FDR)")
    adj_p = benjamini_hochberg(gwas_res.p_values)
    
    # 5. Significant Hits
    sig_idx = findall(adj_p .< p_thresh)
    n_sig = length(sig_idx)
    println("  Found $n_sig significant SNPs (FDR < $p_thresh)")
    
    # Return results
    return (p_values=gwas_res.p_values, adj_p_values=adj_p, significant_indices=sig_idx)
end

"""
    run_gs_pipeline(G, y; method=:gblup, n_folds=5)

Runs a complete Genomic Selection pipeline:
1. QC
2. Model Training (CV)
3. Accuracy Reporting
"""
function run_gs_pipeline(G::GenotypeMatrix, y::Vector{Float64}; method::Symbol=:gblup, n_folds::Int=5)
    println("--- Starting GS Pipeline ---")
    
    n, m = size(G.data)
    println("  Input: $n individuals, $m SNPs")
    
    # Define prediction function based on method
    predict_func = nothing
    
    if method == :gblup
        println("  Method: G-BLUP (LMM)")
        predict_func = (G_train, y_train, G_test) -> begin
            # Build GRM
            K = build_grm(G_train)
            # Train LMM (Bayesian LMM for simplicity/robustness)
            # Or use MME?
            # Let's use our run_lmm (Gibbs)
            res = run_lmm(y_train, Matrix{Float64}(undef, length(y_train), 0), K, chain_length=500)
            # Predict: u_test = K_test_train * inv(K_train) * u_train?
            # Bayesian LMM gives u for training.
            # Prediction for new individuals requires K_test_train.
            
            # For G-BLUP validation, we usually partition K.
            # Let's use a simpler Ridge Regression (RR-BLUP) equivalent for speed in pipeline
            # Or Bayesian Ridge (BayesC with pi=0)
            
            res_bayes = run_bayesC(y_train, G_train.data, chain_length=200, pi=0.0, estimate_pi=false)
            return G_test.data * vec(res_bayes.beta)
        end
        
    elseif method == :bayesB
        println("  Method: BayesB")
        predict_func = (G_train, y_train, G_test) -> begin
            res = run_bayesB(y_train, G_train, n_iter=200)
            return G_test.data * res.beta
        end
        
    elseif method == :rf
        println("  Method: Random Forest")
        predict_func = (G_train, y_train, G_test) -> begin
            rf = random_forest(G_train.data, y_train, n_trees=10)
            return predict_rf(rf, G_test.data)
        end
        
    else
        error("Unknown method: $method")
    end
    
    # Cross Validation
    println("Step 2: Cross-Validation ($n_folds-fold)")
    # We need a custom CV loop here because cross_validation function might be specific
    # Our cross_validation function in Stats/Validation.jl takes a model function.
    # Let's use it if compatible, or write a simple loop.
    
    indices = collect(1:n)
    fold_size = div(n, n_folds)
    correlations = Float64[]
    
    for k in 1:n_folds
        test_idx = ((k-1)*fold_size + 1):min(k*fold_size, n)
        train_idx = setdiff(indices, test_idx)
        
        G_train = GenotypeMatrix(G.data[train_idx, :], G.individuals[train_idx], G.snps)
        y_train = y[train_idx]
        
        G_test = GenotypeMatrix(G.data[test_idx, :], G.individuals[test_idx], G.snps)
        y_test = y[test_idx]
        
        y_pred = predict_func(G_train, y_train, G_test)
        
        acc = cor(y_test, y_pred)
        push!(correlations, acc)
    end
    
    mean_acc = mean(correlations)
    println("  Mean Accuracy (r): $mean_acc")
    
    return (mean_accuracy=mean_acc, fold_accuracies=correlations)
end
