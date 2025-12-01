using Documenter
using AdvancedGenomics

makedocs(
    sitename = "AdvancedGenomics.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    modules = [AdvancedGenomics],
    pages = [
        "Home" => "index.md",
        "About" => "about.md",
        "User Manual" => "manual.md",
        "Tutorials" => "tutorials.md",
        "API Reference" => "api.md"
    ]
)

# deploydocs(
#     repo = "github.com/1958126580/AdvancedGenomics.jl.git",
#     devbranch = "main"
# )
