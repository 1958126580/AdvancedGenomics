using Test
using AdvancedGenomics
using Lux
using Random
using Statistics

# Try to load CUDA
const CUDA_AVAILABLE = try
    using CUDA
    true
catch
    false
end

@testset "GPU Acceleration" begin
    if !CUDA_AVAILABLE || !CUDA.functional()
        @info "CUDA not available or functional. Skipping GPU tests."
        return
    end
    
    @info "Running GPU tests..."
    
    @testset "GWAS GPU" begin
        n = 100
        m = 1000
        G = GenotypeMatrix(rand(0.0:2.0, n, m))
        y = randn(n)
        
        p_vals = run_gwas_gpu(y, G)
        @test length(p_vals) == m
        @test all(0.0 .<= p_vals .<= 1.0)
    end
    
    @testset "GRM GPU" begin
        n = 100
        m = 1000
        G = GenotypeMatrix(rand(0.0:2.0, n, m))
        
        K = build_grm_gpu(G)
        @test size(K) == (n, n)
        @test isapprox(K, K') # Symmetric
    end
    
    @testset "Deep Learning GPU" begin
        model = GenomicTransformer(vocab_size=3, embed_dim=8, num_heads=2, hidden_dim=16, num_layers=1)
        ps, st = Lux.setup(Random.default_rng(), model)
        
        # Move to GPU
        ps_gpu = ps |> ComponentArray |> CuArray
        st_gpu = st # State might need move if it has arrays, but usually empty for this model
        
        # Input
        x = rand(1:3, 10, 2)
        x_gpu = CuArray(x)
        
        # Forward pass
        y_gpu, _ = model(x_gpu, ps_gpu, st_gpu)
        
        @test y_gpu isa CuArray
        @test size(y_gpu) == (1, 2)
    end
end
