# ag.jl - AdvancedGenomics CLI Entry Point

# Activate project environment
import Pkg
Pkg.activate(".")

using AdvancedGenomics
# Run CLI
AdvancedGenomics.CLI.command_main()
