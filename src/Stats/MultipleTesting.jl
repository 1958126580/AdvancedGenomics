"""
    MultipleTesting.jl

Multiple testing correction methods.
"""

using Statistics

"""
    benjamini_hochberg(p_values)

Benjamini-Hochberg FDR correction.
"""
function benjamini_hochberg(p_values::Vector{Float64})
    n = length(p_values)
    idx = sortperm(p_values)
    p_sorted = p_values[idx]
    adj_p = zeros(n)
    
    min_val = 1.0
    for i in n:-1:1
        val = p_sorted[i] * n / i
        min_val = min(min_val, val)
        adj_p[i] = min_val
    end
    
    # Restore order
    res = zeros(n)
    res[idx] = adj_p
    return res
end

"""
    adjust_pvalues(p_values, method)

Adjusts p-values for multiple testing.
Supported methods: "bonferroni", "fdr" (Benjamini-Hochberg).
"""
function adjust_pvalues(p_values::Vector{Float64}, method::String)
    if method == "bonferroni"
        n = length(p_values)
        return min.(p_values .* n, 1.0)
    elseif method == "fdr" || method == "bh"
        return benjamini_hochberg(p_values)
    else
        error("Method $method not supported")
    end
end
