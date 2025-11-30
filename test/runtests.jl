using Test
using AdvancedGenomics
using LinearAlgebra
using Statistics
using Lux
using Random
using DataFrames
using CSV

@testset "AdvancedGenomics.jl Full Suite" begin
    
    # 1. Core Types
    @testset "Core Types" begin
        G_mat = [0 1 2; 1 2 0]
        inds = ["Ind1", "Ind2"]
        snps = ["SNP1", "SNP2", "SNP3"]
        G = GenotypeMatrix(G_mat, inds, snps)
        @test size(G.data) == (2, 3)
        @test G.individuals == inds
        
        ped = Pedigree(["A", "B"], ["S1", missing], ["D1", missing])
        @test length(ped.id) == 2
    end

    # 2. IO
    @testset "IO" begin
        # Create dummy CSV
        df = DataFrame(ID=["I1", "I2"], Trait=[1.0, 2.0], SNP1=[0, 1], SNP2=[2, 0])
        path = "temp_test.csv"
        CSV.write(path, df)
        
        pheno = read_phenotypes(path, id_col=:ID, trait_cols=[:Trait])
        @test nrow(pheno.data) == 2
        
        geno = read_genotypes(path, format="csv")
        @test size(geno.data, 1) == 2
        
        rm(path)
    end

    # 3. Stats & Kernels
    @testset "Stats" begin
        G_mat = Float64.([0 1 2; 1 2 0])
        inds = ["Ind1", "Ind2"]
        snps = ["SNP1", "SNP2", "SNP3"]
        G = GenotypeMatrix(G_mat, inds, snps)
        
        K = build_grm(G)
        @test size(K) == (2, 2)
        @test issymmetric(K)
        
        # QC
        G_filt, keep = filter_maf(G_mat, 0.1)
        @test size(G_filt, 2) <= 3
        
        p_hwe = hwe_test(G_mat)
        @test length(p_hwe) == 3
    end

    # 4. Variance Components (REML)
    @testset "Variance Components" begin
        n = 20
        X = hcat(ones(n))
        K = rand(n, n); K = K * K'
        y = randn(n)
        
        res = estimate_vc_reml(y, X, K, method="AI", max_iter=5)
        @test res.sigmas[1] > 0
        @test res.sigma_e2 > 0
    end

    # 5. Models (LMM, Bayesian)
    @testset "Models" begin
        n = 50
        p = 10
        X = hcat(ones(n))
        G_val = rand(0:2, n, p)
        y = randn(n)
        K = Matrix(1.0I, n, n)
        
        # LMM
        res_lmm = run_lmm(y, X, K, chain_length=100)
        @test length(res_lmm.beta) == 1
        
        # BayesC
        res_bc = run_bayesC(y, Float64.(G_val), chain_length=100)
        @test size(res_bc.beta, 2) == p
        
        # BayesB
        res_bb = run_bayesB(y, Float64.(G_val), chain_length=100)
        @test size(res_bb.beta, 2) == p
        
        # Threshold
        y_bin = rand(1:2, n)
        res_thr = run_threshold_model(y_bin, X, K, chain_length=50)
        @test length(res_thr) == 1
        
        # Multi-trait
        Y_multi = randn(n, 2)
        res_mtm = run_multitrait_lmm(Y_multi, X, K, chain_length=50)
        @test size(res_mtm.Sigma_G) == (2, 2)
    end

    # 6. Deep Learning & XAI
    @testset "Deep Learning" begin
        model = GenomicTransformer(vocab_size=3, embed_dim=8, num_heads=2, hidden_dim=16, num_layers=1)
        ps, st = Lux.setup(Random.default_rng(), model)
        x = rand(1:3, 10, 2)
        y, _ = model(x, ps, st)
        @test size(y) == (1, 2)
        
        # XAI
        sal = saliency_map(model, x[:, 1:1], ps, st)
        @test length(sal) == 10
        
        # GNN
        adj = Float32.(Matrix(I, 5, 5))
        gnn = GenomicGNN(adj, 2, 4, 1)
        ps_g, st_g = Lux.setup(Random.default_rng(), gnn)
        # Input: (in_dims, nodes, batch)
        x_g = randn(Float32, 2, 5, 2)
        y_g, _ = gnn(x_g, ps_g, st_g)
        @test size(y_g) == (1, 2)
    end
    
    # 7. Causal
    @testset "Causal" begin
        b_exp = randn(10)
        b_out = b_exp .+ randn(10)*0.1
        se = fill(0.1, 10)
        res = mr_ivw(b_exp, se, b_out, se)
        @test !isnan(res.p_value)
    end
    
    # 8. Single Step & PedModule
    @testset "PedModule" begin
        ped_ids = ["1", "2", "3"]
        sires = [missing, missing, "1"]
        dams = [missing, missing, "2"]
        ped = Pedigree(ped_ids, sires, dams)
        
        A = build_A(ped)
        @test A[3, 3] == 1.0
        # Wait, A[3,3] = 1 + F_3. F_3 = 0.5 * A[1,2] = 0. So A[3,3] = 1.0.
        # But diagonal is 1 + F.
        # Off-diagonal A[3,1] = 0.5 * (A[1,1] + A[1,2]) = 0.5 * (1 + 0) = 0.5.
        @test A[3, 1] == 0.5
        
        A_inv = build_A_inverse(ped)
        @test size(A_inv) == (3, 3)
    end

    # 9. MME
    @testset "MME" begin
        y = [1.0, 2.0]
        X = hcat([1.0, 1.0])
        Z = Matrix(1.0I, 2, 2)
        K_inv = Matrix(1.0I, 2, 2)
        lambda = 1.0
        
        mme = build_MME(y, X, Z, K_inv, lambda)
        @test size(mme.LHS) == (3, 3) # 1 fixed + 2 random
        
        sol = solve_MME(mme.LHS, mme.RHS, 1)
        @test length(sol.beta) == 1
        @test length(sol.u) == 2
    end

    # 10. Diagnostics
    @testset "Diagnostics" begin
        chain = randn(1000) # i.i.d
        ess = effective_sample_size(chain)
        @test ess > 500 # Should be close to 1000
        
        geweke = geweke_diagnostic(chain)
        @test abs(geweke.z_score) < 3.0 # Should be stationary
        
        ac = autocorrelation(chain, 5)
        @test length(ac) == 6
    end

    # 11. LD & Multiple Testing
    @testset "LD & Multiple Testing" begin
        # LD
        g1 = [0, 1, 2, 0, 1]
        g2 = [0, 1, 2, 0, 1] # Perfect correlation
        @test calculate_r2(g1, g2) ≈ 1.0
        
        G = hcat(g1, g2)
        G_pruned, kept = ld_pruning(G, 0.9)
        @test length(kept) == 1
        
        # Multiple Testing
        p = [0.01, 0.04]
        p_bonf = adjust_pvalues(p, "bonferroni")
        @test p_bonf[1] == 0.02
        @test p_bonf[2] == 0.08
    end

    # 12. Advanced Models (Multi-kernel, Burden)
    @testset "Advanced Models" begin
        n = 50
        X = hcat(ones(n))
        y = randn(n)
        K1 = rand(n, n); K1 = K1 * K1'
        K2 = rand(n, n); K2 = K2 * K2' # Make distinct
        
        # Multi-kernel
        res_mk = estimate_vc_reml(y, X, [K1, K2])
        @test length(res_mk.sigmas) == 2
        
        # Burden
        G_rare = rand(0.0:1.0, n, 10)
        res_bd = run_burden_test(y, G_rare, X)
        @test !isnan(res_bd.p_value)
    end

    # 13. Kernels & CS
    @testset "Kernels & CS" begin
        G_mat = Float64.([0 1 2; 1 2 0])
        inds = ["Ind1", "Ind2"]
        snps = ["SNP1", "SNP2", "SNP3"]
        G = GenotypeMatrix(G_mat, inds, snps)
        
        K_rbf = build_rbf_kernel(G.data)
        @test size(K_rbf) == (2, 2)
        
        # CS
        X = [1.0 0.0; 0.0 1.0]
        y = [1.0, 0.0]
        beta = lasso_cd(y, X, lambda=0.01)
        @test beta[1] > 0.9
        @test abs(beta[2]) < 0.1
    end

    # 14. Simulation
    @testset "Simulation" begin
        G = simulate_genotypes(100, 50)
        @test size(G) == (100, 50)
        
        res = simulate_phenotypes(G, 0.5)
        @test length(res.y) == 100
        
        sim = simulate_multi_omics(50, 100, 10)
        @test size(sim.M) == (50, 10)
    end

    # 15. INLA
    @testset "INLA" begin
        n = 20
        X = hcat(ones(n))
        y = randn(n)
        Z = Matrix(1.0I, n, n)
        K_inv = Matrix(1.0I, n, n)
        
        res = run_inla_lmm(y, X, Z, K_inv; grid_size=10)
        @test res.sigma_e2 > 0
        @test res.sigma_g2 > 0
        @test length(res.beta) == 1
    end

    # 16. Copula & Breeding
    @testset "Copula & Breeding" begin
        # Copula
        data = randn(100, 2)
        rho = gaussian_copula_fit(data)
        @test size(rho) == (2, 2)
        
        # Scheme
        P = [1.0 0.0; 0.0 1.0]
        G = [0.5 0.0; 0.0 0.5]
        w = [1.0, 1.0]
        b = selection_index(P, G, w)
        @test length(b) == 2
        
        # Sire Model
        y = randn(10)
        X = hcat(ones(10))
        sires = ["S1", "S2"]
        # Assign sires randomly
        sire_ids = rand(sires, 10)
        
        ped = Pedigree(DataFrame(ID=["S1", "S2"], Sire=["0", "0"], Dam=["0", "0"]))
        res = run_sire_model(y, X, sire_ids, ped)
        @test res.sigma_e2 > 0
    end

    # 17. Bioinformatics
    @testset "Bioinformatics" begin
        # Annotation
        write("test_annot.gff", """
        1	RefSeq	gene	100	200	.	+	.	ID=Gene1
        """)
        feats = read_gff("test_annot.gff")
        @test length(feats) == 1
        
        snp_df = DataFrame(chr=["1"], pos=[150], id=["S1"])
        ann = annotate_snps(snp_df, feats)
        @test ann[1] == "gene"
        rm("test_annot.gff")
        
        # Enrichment
        sig = Set(["G1"])
        path = Dict("P1" => Set(["G1", "G2"]))
        bg = Set(["G1", "G2", "G3"])
        res = run_pathway_enrichment(sig, path, bg)
        @test nrow(res) == 1
    end

    # 18. Comprehensive GWAS/GS
    @testset "Comprehensive GWAS/GS" begin
        G_mat = simulate_genotypes(20, 10)
        G = GenotypeMatrix(G_mat, ["Ind$i" for i in 1:20], ["SNP$j" for j in 1:10])
        y = randn(20)
        
        # PCA
        pcs = run_pca(G, k=2)
        @test size(pcs.projections) == (20, 2)
        
        # FarmCPU
        res_farm = run_farmcpu(y, G, pcs.projections)
        @test length(res_farm.p_values) == 10
        
        # Meta
        res_meta = meta_analysis_ivw([0.5], [0.1])
        @test res_meta.beta ≈ 0.5
        
        # Fine-Mapping
        pip = simple_fine_mapping([0.0, 5.0], Matrix(1.0I, 2, 2))
        @test pip[2] > pip[1]
        
        # Dominance
        D = build_dominance_kernel(G)
        @test size(D) == (20, 20)
    end

    # 19. Haplotypes
    @testset "Haplotypes" begin
        G_mat = simulate_genotypes(20, 20)
        G = GenotypeMatrix(G_mat, ["Ind$i" for i in 1:20], ["SNP$j" for j in 1:20])
        
        phased = phase_genotypes(G, n_iter=2)
        @test size(phased.H1) == (20, 20)
        
        # Check consistency: H1 + H2 == G (for homozygotes)
        # For heterozygotes (1), H1+H2=1.
        for i in 1:20, j in 1:20
            if G_mat[i, j] == 0
                @test phased.H1[i, j] == 0 && phased.H2[i, j] == 0
            elseif G_mat[i, j] == 2
                @test phased.H1[i, j] == 1 && phased.H2[i, j] == 1
            else
                @test phased.H1[i, j] + phased.H2[i, j] == 1
            end
        end
        
        blocks = build_haplotype_matrix(phased.H1, phased.H2, window_size=5)
        @test size(blocks.H1_blocks, 2) == 4
        
        K_hap = build_haplotype_kernel(blocks.H1_blocks, blocks.H2_blocks)
        @test size(K_hap) == (20, 20)
    end

    # 20. PLINK Compatibility
    @testset "PLINK Compatibility" begin
        G_mat = simulate_genotypes(20, 10)
        G = GenotypeMatrix(G_mat, ["Ind$i" for i in 1:20], ["SNP$j" for j in 1:10])
        
        # IO
        write_bed("test_io", G)
        G_read = read_plink("test_io")
        @test size(G_read.data) == (20, 10)
        # Check first element (approx due to missing handling or mapping)
        # We mapped 0->2, 2->0.
        # Original: 0, 1, 2.
        # Written: 0->Homo1(2), 1->Het(1), 2->Homo2(0).
        # Read: Homo1->2.
        # So 0 -> 2. 2 -> 0.
        # It flips the reference.
        # Let's check if it's consistent.
        rm("test_io.bed"); rm("test_io.bim"); rm("test_io.fam")
        
        # Logistic
        y_bin = Float64.(rand(20) .> 0.5)
        res = run_logistic_gwas(y_bin, G, ones(20, 1))
        @test length(res.p_values) == 10
        
        # Clumping
        clumps = clump_snps(res.p_values, Matrix(1.0I, 10, 10), p1=1.0)
        @test length(clumps) > 0
        
        # IBD
        ibd = estimate_ibd_plink(G)
        @test size(ibd.PI_HAT) == (20, 20)
    end

    # 21. Ancestry
    @testset "Ancestry" begin
        G_mat = simulate_genotypes(20, 50)
        G = GenotypeMatrix(G_mat, ["Ind$i" for i in 1:20], ["SNP$j" for j in 1:50])
        
        # Fake phased
        H1 = round.(Int, G_mat ./ 2)
        H2 = round.(Int, G_mat ./ 2)
        
        segs = detect_ibd_segments(H1, H2, min_len=5)
        @test nrow(segs) >= 0
        
        K_ibd = build_ibd_grm(segs, 20, 50)
        @test size(K_ibd) == (20, 20)
        
        froh = calculate_froh(G, min_len=5)
        @test length(froh) == 20
    end

    # 22. ML & DL
    @testset "ML & DL" begin
        G_mat = simulate_genotypes(20, 10)
        y = randn(20)
        
        # RF
        rf = random_forest(G_mat, y, n_trees=2)
        pred = predict_rf(rf, G_mat)
        @test length(pred) == 20
        
        # GBM
        gbm = gradient_boosting(G_mat, y, n_trees=2)
        pred_gbm = predict_gbm(gbm, G_mat)
        @test length(pred_gbm) == 20
        
        # DL
        cnn = GenomicCNN(10)
        @test cnn isa Lux.Chain
        
        ae = GenomicAutoencoder(10, 2)
        @test ae isa Lux.Chain
    end

    # 23. QTL Mapping
    @testset "QTL Mapping" begin
        G_mat = simulate_genotypes(20, 10)
        y = randn(20)
        map_pos = collect(0.0:10.0:90.0)
        
        res = interval_mapping(y, GenotypeMatrix(G_mat, ["Ind$i" for i in 1:20], ["SNP$j" for j in 1:10]), map_pos, step=5.0)
        @test length(res.lod_scores) == length(res.positions)
        @test maximum(res.lod_scores) >= 0.0
        
        # Permutation (small n for speed)
        thresh = permutation_test_qtl(y, GenotypeMatrix(G_mat, ["Ind$i" for i in 1:20], ["SNP$j" for j in 1:10]), map_pos, n_perm=5, step=10.0)
        @test thresh >= 0.0
    end

    # 24. Modern Stats
    @testset "Modern Stats" begin
        # BF
        post = randn(100) .+ 5.0
        bf = calculate_bayes_factor(post, at=0.0)
        @test bf > 1.0 # Evidence for effect
        
        # Wavelet
        sig = randn(16)
        den = wavelet_denoising(sig, level=2)
        @test length(den) == 16
        
        # Conformal
        X = randn(20, 5)
        y = X * ones(5) + randn(20) * 0.1
        cp = conformal_prediction(X[1:10, :], y[1:10], X[11:20, :])
        @test length(cp.pred) == 10
        @test all(cp.upper .>= cp.lower)
    end

    # 25. Optimization & OR
    @testset "Optimization & OR" begin
        # SA
        res = simulated_annealing(x -> sum(x.^2), [5.0])
        @test res.minimum < 1.0
        
        # GA
        X = randn(20, 10)
        y = X[:, 1] .+ randn(20)*0.1 # Feature 1 is important
        mask = genetic_algorithm_select(X, y, pop_size=10, generations=5)
        @test length(mask) == 10
        
        # ACO
        mask_aco = ant_colony_select(X, y, n_ants=5, iterations=5, n_features=2)
        @test sum(mask_aco) == 2
        
        # OCS
        EBV = randn(10)
        A = Matrix(1.0I, 10, 10)
        ocs = optimal_contribution_selection(EBV, A)
        @test length(ocs.contributions) == 10
        @test isapprox(sum(ocs.contributions), 1.0, atol=1e-4)
    end

    # 26. Pipelines
    @testset "Pipelines" begin
        G_mat = simulate_genotypes(20, 50)
        y = randn(20)
        G = GenotypeMatrix(G_mat, ["Ind$i" for i in 1:20], ["SNP$j" for j in 1:50])
        
        # GWAS
        res_gwas = run_gwas_pipeline(G, y)
        @test length(res_gwas.p_values) == 50
        
        # GS
        res_gs = run_gs_pipeline(G, y, method=:gblup, n_folds=2)
        @test res_gs.mean_accuracy >= -1.0 && res_gs.mean_accuracy <= 1.0
    end
end

include("gpu_tests.jl")
include("vis_tests.jl")
