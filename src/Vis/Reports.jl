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

"""
    generate_gs_html_report(results, filename::String; title="Genomic Selection Report", method="gblup")

Generates a standalone HTML report for Genomic Selection results.
Results should be a NamedTuple with fields: mean_accuracy, fold_accuracies.
"""
function generate_gs_html_report(results, filename::String; title="Genomic Selection Report", method="gblup")
    # Ensure filename ends with .html
    if !endswith(filename, ".html")
        filename *= ".html"
    end
    
    # Extract results
    mean_acc = results.mean_accuracy
    fold_accs = results.fold_accuracies
    n_folds = length(fold_accs)
    
    # Create accuracy plot
    img_path = replace(filename, ".html" => "_accuracy.png")
    
    # Bar chart of fold accuracies
    plt = bar(1:n_folds, fold_accs, 
              title="Cross-Validation Accuracy by Fold",
              xlabel="Fold", 
              ylabel="Accuracy (Correlation)",
              legend=false,
              color=:steelblue,
              ylims=(min(0, minimum(fold_accs) - 0.1), max(1, maximum(fold_accs) + 0.1)))
    
    # Add mean line
    hline!(plt, [mean_acc], color=:red, linewidth=2, linestyle=:dash, label="Mean")
    
    savefig(plt, img_path)
    
    # Calculate statistics
    std_acc = length(fold_accs) > 1 ? std(fold_accs) : 0.0
    min_acc = minimum(fold_accs)
    max_acc = maximum(fold_accs)
    
    # HTML Template
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>$title</title>
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                margin: 40px;
                background: #f8f9fa;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 40px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            h1 { 
                color: #2c3e50; 
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }
            h2 { 
                color: #34495e; 
                margin-top: 30px;
                border-left: 4px solid #3498db;
                padding-left: 10px;
            }
            .metric-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .metric-card.primary {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            }
            .metric-label {
                font-size: 14px;
                opacity: 0.9;
                margin-bottom: 5px;
            }
            .metric-value {
                font-size: 32px;
                font-weight: bold;
            }
            .plot-container { 
                margin: 20px 0;
                text-align: center;
            }
            img { 
                max-width: 100%; 
                height: auto;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            table { 
                border-collapse: collapse; 
                width: 100%;
                margin: 20px 0;
            }
            th, td { 
                text-align: left; 
                padding: 12px; 
                border-bottom: 1px solid #ddd; 
            }
            th {
                background: #3498db;
                color: white;
                font-weight: 600;
            }
            tr:hover { background-color: #f5f5f5; }
            .info-box {
                background: #e8f4f8;
                border-left: 4px solid #3498db;
                padding: 15px;
                margin: 20px 0;
                border-radius: 4px;
            }
            .footer {
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                text-align: center;
                color: #7f8c8d;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧬 $title</h1>
            <p style="color: #7f8c8d;">Generated on: $(now())</p>
            
            <div class="info-box">
                <strong>Method:</strong> $(uppercase(string(method))) | 
                <strong>Cross-Validation:</strong> $n_folds-Fold
            </div>
            
            <h2>📊 Performance Metrics</h2>
            <div class="metric-grid">
                <div class="metric-card primary">
                    <div class="metric-label">Mean Accuracy</div>
                    <div class="metric-value">$(round(mean_acc, digits=4))</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Std. Deviation</div>
                    <div class="metric-value">$(round(std_acc, digits=4))</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Min Accuracy</div>
                    <div class="metric-value">$(round(min_acc, digits=4))</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Max Accuracy</div>
                    <div class="metric-value">$(round(max_acc, digits=4))</div>
                </div>
            </div>
            
            <h2>📈 Cross-Validation Results</h2>
            <div class="plot-container">
                <img src="$(basename(img_path))" alt="Accuracy Plot">
            </div>
            
            <h2>📋 Detailed Results</h2>
            <table>
                <tr>
                    <th>Fold</th>
                    <th>Accuracy (Correlation)</th>
                    <th>Performance</th>
                </tr>
                $(join([
                    let
                        perf = fold_accs[i] >= mean_acc ? "✓ Above Mean" : "Below Mean"
                        "<tr><td>Fold $i</td><td>$(round(fold_accs[i], digits=4))</td><td>$perf</td></tr>"
                    end
                    for i in 1:n_folds
                ], "\n                "))
            </table>
            
            <div class="footer">
                <p>Generated by AdvancedGenomics.jl | 
                   <a href="https://github.com/1958126580/AdvancedGenomics" style="color: #3498db;">GitHub</a>
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    
    open(filename, "w") do io
        write(io, html_content)
    end
    
    @info "GS Report generated: $filename"
end

