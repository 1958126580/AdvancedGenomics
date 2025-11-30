using Test
using AdvancedGenomics
using PlotlyJS

@testset "Visualization" begin
    # Dummy data
    p_values = rand(100)
    chr = rand(1:5, 100)
    pos = collect(1:100)
    
    @testset "Manhattan Interactive" begin
        # Just check if it runs and returns a Plot object
        plt = manhattan_plot_interactive(p_values, chr, pos)
        @test plt !== nothing
        # @test plt isa PlotlyJS.Plot # Returns SyncPlot/WebIO.Scope in some envs
    end
    
    @testset "QQ Interactive" begin
        plt = qq_plot_interactive(p_values)
        @test plt !== nothing
    end
end
