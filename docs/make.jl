using Documenter
using AdvancedGenomics

makedocs(
    sitename = "AdvancedGenomics.jl",
    format = Documenter.HTML(
        prettyurls = true,
        edit_link = "main"
    ),
    pages = [
        "Home" => "index.md",
        "Manual" => "manual.md",
        "API" => "api.md"
    ],
    warnonly = true,
    clean = true
)
