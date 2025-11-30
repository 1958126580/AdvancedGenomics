# ==============================================================================
# Run All Examples
# ==============================================================================
# This script executes all example scripts in the directory to verify
# that they run without error.
# ==============================================================================

using Pkg
Pkg.activate(".")

examples = [
    "01_GWAS_Standard.jl",
    "02_GWAS_Advanced.jl",
    "03_GS_Kernels_ML.jl",
    "04_GS_Bayesian.jl",
    "05_Deep_Learning.jl",
    "06_Complex_Models.jl",
    "07_Breeding.jl",
    "08_Multi_Omics.jl",
    "09_Modern_Stats.jl",
    "10_HPC_PopGen.jl"
]

println("--- Starting Verification of All Examples ---")
println("Found $(length(examples)) examples.")

passed = 0
failed = 0

for example in examples
    println("\n" * "="^60)
    println("Running $example...")
    println("="^60)
    
    try
        # Run the script in a separate process or include it
        # Using include() ensures it runs in the current environment
        # We wrap in a module to avoid namespace pollution between scripts
        @eval module $(Symbol("Example_" * replace(example, ".jl" => "")))
            # include is relative to the file executing it
            include($example)
        end
        
        println("\n[SUCCESS] $example passed.")
        global passed += 1
    catch e
        println("\n[FAILURE] $example failed.")
        showerror(stdout, e, catch_backtrace())
        global failed += 1
    end
end

println("\n" * "="^60)
println("Verification Summary")
println("="^60)
println("Total:  $(length(examples))")
println("Passed: $passed")
println("Failed: $failed")

if failed == 0
    println("\nAll examples ran successfully!")
    exit(0)
else
    println("\nSome examples failed.")
    exit(1)
end
