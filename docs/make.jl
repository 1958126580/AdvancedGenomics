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
        "API Reference" => "api.md"
    ]
)

# deploydocs(
#     repo = "github.com/Antigravity/AdvancedGenomics.jl.git",
# )
