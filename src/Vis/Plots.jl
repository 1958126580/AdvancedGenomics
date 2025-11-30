"""
    Plots.jl

Visualization module.
Includes Manhattan and QQ plots using PlotlyJS.
"""

using Statistics
using PlotlyJS

"""
    manhattan_plot(p_values, chr, pos; threshold=5e-8)

Generates a Manhattan plot data structure.
"""
function manhattan_plot(p_values::Vector{Float64}, chr::Vector{Int}, pos::Vector{Int})
    logp = -log10.(p_values)
    return (x=pos, y=logp, chr=chr)
end

"""
    manhattan_plot_interactive(p_values, chr, pos; threshold=5e-8, title="Manhattan Plot")

Generates an interactive Manhattan plot using PlotlyJS.
"""
function manhattan_plot_interactive(p_values::Vector{Float64}, chr::Vector{Int}, pos::Vector{Int}; threshold::Float64=5e-8, title::String="Manhattan Plot")
    logp = -log10.(p_values)
    
    # Create traces for each chromosome for color differentiation
    unique_chr = sort(unique(chr))
    traces = PlotlyJS.GenericTrace[]
    
    # Simple color cycle
    colors = ["#1f77b4", "#ff7f0e"]
    
    for (i, c) in enumerate(unique_chr)
        idx = chr .== c
        trace = PlotlyJS.scatter(
            x = pos[idx],
            y = logp[idx],
            mode = "markers",
            name = "Chr $c",
            marker = PlotlyJS.attr(color = colors[(i % 2) + 1], size=6),
            text = ["Chr: $c, Pos: $(p)" for p in pos[idx]]
        )
        push!(traces, trace)
    end
    
    # Threshold line
    line = PlotlyJS.attr(
        type="line",
        x0=minimum(pos),
        x1=maximum(pos),
        y0=-log10(threshold),
        y1=-log10(threshold),
        line=PlotlyJS.attr(color="red", dash="dash")
    )
    
    layout = PlotlyJS.Layout(
        title=title,
        xaxis=PlotlyJS.attr(title="Position"),
        yaxis=PlotlyJS.attr(title="-log10(P-value)"),
        shapes=[line]
    )
    
    return PlotlyJS.plot(traces, layout)
end

"""
    qq_plot(p_values)

Generates QQ plot data.
"""
function qq_plot(p_values::Vector{Float64})
    n = length(p_values)
    observed = sort(-log10.(p_values), rev=true)
    expected = -log10.(range(1/n, 1.0, length=n))
    return (expected=expected, observed=observed)
end

"""
    qq_plot_interactive(p_values; title="QQ Plot")

Generates an interactive QQ plot using PlotlyJS.
"""
function qq_plot_interactive(p_values::Vector{Float64}; title::String="QQ Plot")
    data = qq_plot(p_values)
    
    trace = PlotlyJS.scatter(
        x = data.expected,
        y = data.observed,
        mode = "markers",
        name = "Observed"
    )
    
    # Diagonal line
    max_val = max(maximum(data.expected), maximum(data.observed))
    line = PlotlyJS.attr(
        type="line",
        x0=0,
        x1=max_val,
        y0=0,
        y1=max_val,
        line=PlotlyJS.attr(color="red", dash="dash")
    )
    
    layout = PlotlyJS.Layout(
        title=title,
        xaxis=PlotlyJS.attr(title="Expected -log10(P)"),
        yaxis=PlotlyJS.attr(title="Observed -log10(P)"),
        shapes=[line]
    )
    
    return PlotlyJS.plot([trace], layout)
end
