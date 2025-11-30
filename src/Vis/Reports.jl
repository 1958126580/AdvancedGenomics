using Plots
using Dates

"""
    generate_html_report(results, filename::String; title="Genomic Analysis Report")

Generates a standalone HTML report containing summary statistics and plots.
"""
function generate_html_report(results, filename::String; title="Genomic Analysis Report")
    # Ensure filename ends with .html
    if !endswith(filename, ".html")
        filename *= ".html"
    end

    # Create plots
    # Mock chr/pos for demo if not present
    n = length(results.p_values)
    chr = ones(Int, n)
    pos = collect(1:n)
    
    p1 = manhattan_plot(results.p_values, chr, pos)
    p2 = qq_plot(results.p_values)
    
    img1_path = replace(filename, ".html" => "_manhattan.png")
    img2_path = replace(filename, ".html" => "_qq.png")
    
    # Real Plotting
    plt1 = scatter(p1.x, p1.y, title="Manhattan Plot", xlabel="Position", ylabel="-log10(P)", legend=false)
    savefig(plt1, img1_path)
    
    plt2 = scatter(p2.expected, p2.observed, title="QQ Plot", xlabel="Expected", ylabel="Observed", legend=false)
    plot!(plt2, [0, maximum(p2.expected)], [0, maximum(p2.expected)], color=:red)
    savefig(plt2, img2_path)
    
    # Basic HTML Template
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>$title</title>
        <style>
            body { font-family: sans-serif; margin: 40px; }
            h1 { color: #333; }
            .container { display: flex; flex-wrap: wrap; }
            .plot { margin: 20px; border: 1px solid #ddd; padding: 10px; border-radius: 5px; }
            img { max-width: 100%; height: auto; }
            table { border-collapse: collapse; width: 100%; }
            th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
            tr:hover { background-color: #f5f5f5; }
        </style>
    </head>
    <body>
        <h1>$title</h1>
        <p>Generated on: $(now())</p>
        
        <h2>Summary Statistics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Markers</td><td>$(length(results.p_values))</td></tr>
            <tr><td>Significant Markers (P < 5e-8)</td><td>$(count(x -> x < 5e-8, results.p_values))</td></tr>
        </table>
        
        <h2>Visualizations</h2>
        <div class="container">
            <div class="plot">
                <h3>Manhattan Plot</h3>
                <img src="$(basename(img1_path))" alt="Manhattan Plot">
            </div>
            <div class="plot">
                <h3>QQ Plot</h3>
                <img src="$(basename(img2_path))" alt="QQ Plot">
            </div>
        </div>
        
        <h2>Top Hits</h2>
        <table>
            <tr><th>Rank</th><th>P-Value</th></tr>
            $(join([ "<tr><td>$i</td><td>$(results.p_values[i])</td></tr>" for i in sortperm(results.p_values)[1:min(10, end)] ], "\n"))
        </table>
    </body>
    </html>
    """
    
    open(filename, "w") do io
        write(io, html_content)
    end
    
    @info "Report generated: $filename"
end
