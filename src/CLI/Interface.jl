module CLI

using Comonicon
using Term
using ..AdvancedGenomics


"""
    gwas(; geno, pheno, out="gwas_result", report=false)

Run Genome-Wide Association Study.

# Args
- `geno`: Path to genotype file (CSV).
- `pheno`: Path to phenotype file (CSV).
- `out`: Output prefix.

# Flags
- `--report`: Generate HTML report.
"""
@cast function gwas(; geno::String, pheno::String, out::String="gwas_result", report::Bool=false)
    tprint(Panel("{bold green}Running GWAS{/bold green}", title="AdvancedGenomics", style="green"))
    
    if !isfile(geno) || !isfile(pheno)
        # Simulation mode for demo if files missing
        @warn "Files not found. Simulating data for demonstration..."
        tprint(Panel("Simulating 100 individuals, 1000 SNPs...", title="Simulation", style="yellow"))
        G_mat = simulate_genotypes(100, 1000)
        y = randn(100)
        G = GenotypeMatrix(G_mat, String[], String[])
    else
        # Load actual data files
        @info "Loading genotype data from $geno..."
        G = read_genotypes(geno, format="csv")
        
        @info "Loading phenotype data from $pheno..."
        pheno_data = read_phenotypes(pheno, id_col=:ID, trait_cols=[:Trait])
        y = pheno_data.data.Trait
        
        # Ensure dimensions match
        if size(G.data, 1) != length(y)
            error("Dimension mismatch: $(size(G.data, 1)) individuals in genotype vs $(length(y)) in phenotype")
        end
    end

    # Progress bar example
    pbar = ProgressBar(columns=:bar)
    job = addjob!(pbar; N=100)
    start!(pbar)
    for i in 1:100
        update!(job)
        sleep(0.01)
    end
    stop!(pbar)

    @info "Running Pipeline..."
    res = run_gwas_pipeline(G, y)
    
    @info "Saving results to $out.txt..."
    open(out * ".txt", "w") do io
        println(io, "SNP_Index\tP_Value\tAdj_P_Value")
        for i in 1:length(res.p_values)
            println(io, "$i\t$(res.p_values[i])\t$(res.adj_p_values[i])")
        end
    end

    if report
        @info "Generating HTML Report..."
        generate_html_report(res, out * "_report.html", title="GWAS Analysis Report")
        tprint(Panel("Report saved to {bold}$out_report.html{/bold}", style="blue"))
    end

    tprint(Panel("{bold green}Done!{/bold green}", style="green"))
end

"""
    gs(; geno, pheno, method="gblup", out="gs_result", report=false)

Run Genomic Selection.

# Args
- `geno`: Path to genotype file.
- `pheno`: Path to phenotype file.
- `method`: Method (gblup, bayesB, rf).
- `out`: Output prefix.

# Flags
- `--report`: Generate HTML report.
"""
@cast function gs(; geno::String, pheno::String, method::String="gblup", out::String="gs_result", report::Bool=false)
    tprint(Panel("{bold blue}Running Genomic Selection{/bold blue}", title="AdvancedGenomics", style="blue"))
    
    # Simulation for demo
    @warn "Simulating data..."
    G_mat = simulate_genotypes(100, 500)
    y = randn(100)
    G = GenotypeMatrix(G_mat, String[], String[])
    
    @info "Method: $method"
    res = run_gs_pipeline(G, y, method=Symbol(method))
    
    tprint(Panel("Accuracy: {bold}$(res.mean_accuracy){/bold}", title="Results", style="green"))
    
    if report
        # GS report might be different, but using same for now
        # generate_html_report(res, out * "_report.html") # GS res might not have p-values
        @warn "Report generation for GS not fully implemented yet."
    end
end

"""
    sim(; n=100, m=1000, out="sim_data")

Simulate Genomic Data.

# Args
- `n`: Number of individuals.
- `m`: Number of markers.
- `out`: Output prefix.
"""
@cast function sim(; n::Int=100, m::Int=1000, out::String="sim_data")
    tprint(Panel("Simulating $n individuals, $m SNPs", title="Simulation", style="magenta"))
    G = simulate_genotypes(n, m)
    y = randn(n)
    @info "Data simulated."
end



end # module
