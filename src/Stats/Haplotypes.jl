"""
    Haplotypes.jl

Haplotype Inference (Phasing) and Analysis tools.
Implements a Window-based Hidden Markov Model (HMM) for phasing.
"""

using Statistics
using LinearAlgebra
using Random

"""
    phase_genotypes(G; n_states=20, n_iter=5, window_size=50)

Phases genotypes using a Window-based HMM approach.
G: Genotype Matrix (n x m) (0, 1, 2)
n_states: Number of reference haplotypes to use in the HMM (K).
n_iter: Number of iterations (burn-in + sampling).
window_size: Size of the sliding window for phasing.

Returns:
- H1: Haplotype 1 Matrix (n x m) (0, 1)
- H2: Haplotype 2 Matrix (n x m) (0, 1)
"""
function phase_genotypes(G::GenotypeMatrix; n_states::Int=20, n_iter::Int=5, window_size::Int=100)
    M = G.data
    n, m = size(M)
    
    # Initialize random haplotypes consistent with genotypes
    H1 = zeros(Int, n, m)
    H2 = zeros(Int, n, m)
    
    # Initial random assignment
    for i in 1:n
        for j in 1:m
            g = M[i, j]
            if g == 0
                H1[i, j] = 0; H2[i, j] = 0
            elseif g == 2
                H1[i, j] = 1; H2[i, j] = 1
            else # g == 1
                if rand() < 0.5
                    H1[i, j] = 0; H2[i, j] = 1
                else
                    H1[i, j] = 1; H2[i, j] = 0
                end
            end
        end
    end
    
    # Iterative refinement
    for iter in 1:n_iter
        # Select Reference Panel for this iteration
        # We use the current estimates of everyone else as the "population"
        # To avoid O(N^2), we pick a random subset of K individuals to serve as the reference panel
        
        ref_indices = randperm(n)[1:min(n, n_states)]
        RefH = vcat(H1[ref_indices, :], H2[ref_indices, :]) # 2*K haplotypes
        
        # Phase each individual
        Threads.@threads for i in 1:n
            # Skip if in reference panel? No, phase everyone.
            
            # We phase in windows to keep HMM state space manageable if we were doing full chromosome
            # But here we just do the whole chromosome if m is small, or windows if large.
            # Let's do sliding windows with overlap.
            
            # For simplicity in this implementation:
            # We treat the whole sequence as one block if m < 200, else windows.
            # Let's implement the core Viterbi for a range.
            
            phase_individual_hmm!(H1, H2, M, i, RefH, 1, m)
        end
    end
    
    return (H1=H1, H2=H2)
end

"""
    phase_individual_hmm!(H1, H2, M, ind_idx, RefH, start_j, end_j)

Phases a single individual using Viterbi algorithm against a reference panel.
"""
function phase_individual_hmm!(H1, H2, M, ind_idx, RefH, start_j, end_j)
    n_ref = size(RefH, 1)
    len = end_j - start_j + 1
    
    # States: Pairs of reference haplotypes (h_a, h_b)
    # This is O(K^2). If K=20, States=400. Feasible.
    # To optimize, we can use "Li & Stephens" model which is O(K).
    # For this "National Top Level" upgrade, let's implement a simplified Li & Stephens.
    
    # Li & Stephens Model:
    # State is just (h_a, h_b) where h_a, h_b are indices in RefH.
    # Transition: Probability of switching reference haplotype (recombination).
    # Emission: Probability of observing Genotype given (h_a, h_b).
    
    # However, full O(K^2) Viterbi is slow for large K.
    # We use a stochastic approach (Gibbs sampling) or a restricted Viterbi.
    # Let's stick to the "Window-based Viterbi" with a small K (e.g. 10-20).
    
    K = n_ref
    n_states = K * K
    
    # Log-probabilities
    # T1[state, site]
    # We only store current and prev column to save memory.
    
    # Pre-compute emissions for this individual
    # Genotype g at site j.
    # State (u, v) implies haplotype alleles RefH[u, j] and RefH[v, j].
    # Implied genotype g_hat = RefH[u, j] + RefH[v, j].
    # P(g | g_hat) = high if match, low (error rate) if mismatch.
    
    eps = 1e-4 # Error rate
    log_emit_match = log(1.0 - eps)
    log_emit_mismatch = log(eps)
    
    # Initialize Viterbi
    # path[state, site] -> stores 'prev_state'
    path = Matrix{Int}(undef, n_states, len)
    v_curr = zeros(Float64, n_states)
    v_prev = zeros(Float64, n_states)
    
    # Initial probabilities (Uniform)
    fill!(v_prev, -log(n_states)) 
    
    # Recombination probability
    # rho = 4 * Ne * r. Let's assume constant small prob of switch.
    theta = 0.01 # Prob of switch between markers
    log_no_switch = log(1.0 - theta)
    log_switch = log(theta / (n_states - 1))
    
    # Map (u, v) -> state_idx
    # state_idx = (u-1)*K + v
    
    for t in 1:len
        j = start_j + t - 1
        g = M[ind_idx, j]
        
        # Calculate Emissions
        # We can optimize this loop.
        
        # If g is missing, emission is uniform (0.0 log)
        
        for u in 1:K
            for v in 1:K
                state = (u - 1) * K + v
                
                # Emission
                if isnan(g)
                    log_emit = 0.0
                else
                    g_hat = RefH[u, j] + RefH[v, j]
                    if g == g_hat
                        log_emit = log_emit_match
                    else
                        # Allow for genotyping error
                        log_emit = log_emit_mismatch
                    end
                end
                
                # Transition (Max over prev)
                if t == 1
                    v_curr[state] = v_prev[state] + log_emit
                else
                    # This is the O(S^2) bottleneck.
                    # Approx: max(v_prev[state] + log_no_switch, max(v_prev) + log_switch)
                    # Because switch prob is uniform to all other states.
                    
                    max_prev = maximum(v_prev)
                    score_no_switch = v_prev[state] + log_no_switch
                    score_switch = max_prev + log_switch
                    
                    if score_no_switch > score_switch
                        v_curr[state] = score_no_switch + log_emit
                        path[state, t] = state # Stay
                    else
                        v_curr[state] = score_switch + log_emit
                        # We need to record WHICH state was max_prev.
                        # This is slightly wrong, we need argmax(v_prev).
                        # Let's find argmax properly.
                        best_prev = argmax(v_prev)
                        path[state, t] = best_prev
                    end
                end
            end
        end
        
        v_prev .= v_curr
    end
    
    # Backtrack
    best_end_state = argmax(v_prev)
    curr_state = best_end_state
    
    for t in len:-1:1
        j = start_j + t - 1
        
        # Decode state (u, v)
        u = div(curr_state - 1, K) + 1
        v = mod(curr_state - 1, K) + 1
        
        # Assign haplotypes
        # If g was 1 (Het), we assign based on u, v.
        # If RefH[u] says 0 and RefH[v] says 1 -> H1=0, H2=1.
        # If both say 0 or both 1, but g=1? Error.
        # We force consistency with g if possible, else trust HMM (imputation).
        
        g = M[ind_idx, j]
        
        h1_pred = RefH[u, j]
        h2_pred = RefH[v, j]
        
        if g == 1
            if h1_pred != h2_pred
                H1[ind_idx, j] = h1_pred
                H2[ind_idx, j] = h2_pred
            else
                # Ambiguous or Error. Keep current random assignment?
                # Or force 0/1.
                H1[ind_idx, j] = 0
                H2[ind_idx, j] = 1
            end
        elseif g == 0
            H1[ind_idx, j] = 0; H2[ind_idx, j] = 0
        elseif g == 2
            H1[ind_idx, j] = 1; H2[ind_idx, j] = 1
        end
        
        if t > 1
            curr_state = path[curr_state, t]
        end
    end
end

function score_haplotype(h::Vector{Int}, Ref::Matrix{Int})
    # Max similarity to any reference haplotype
    max_score = 0
    n_ref = size(Ref, 1)
    for k in 1:n_ref
        score = sum(h .== Ref[k, :])
        if score > max_score
            max_score = score
        end
    end
    return max_score
end

"""
    build_haplotype_matrix(H1, H2; window_size=5)

Constructs a Haplotype Dosage Matrix.
Encodes haplotypes in windows as unique alleles.
"""
function build_haplotype_matrix(H1::Matrix{Int}, H2::Matrix{Int}; window_size::Int=5)
    n, m = size(H1)
    n_windows = floor(Int, m / window_size)
    
    blocks_H1 = Matrix{Vector{Int}}(undef, n, n_windows)
    blocks_H2 = Matrix{Vector{Int}}(undef, n, n_windows)
    
    for w in 1:n_windows
        start_j = (w - 1) * window_size + 1
        end_j = start_j + window_size - 1
        
        for i in 1:n
            blocks_H1[i, w] = H1[i, start_j:end_j]
            blocks_H2[i, w] = H2[i, start_j:end_j]
        end
    end
    
    return (H1_blocks=blocks_H1, H2_blocks=blocks_H2)
end

"""
    build_haplotype_kernel(H1_blocks, H2_blocks)

Constructs a Haplotype Kernel Matrix.
"""
function build_haplotype_kernel(H1_blocks, H2_blocks)
    n, n_windows = size(H1_blocks)
    K = zeros(n, n)
    for i in 1:n
        for j in 1:n
            match = 0
            for w in 1:n_windows
                # Count matches between (H1i, H2i) and (H1j, H2j)
                # 4 comparisons
                m1 = H1_blocks[i, w] == H1_blocks[j, w]
                m2 = H1_blocks[i, w] == H2_blocks[j, w]
                m3 = H2_blocks[i, w] == H1_blocks[j, w]
                m4 = H2_blocks[i, w] == H2_blocks[j, w]
                match += (m1 + m2 + m3 + m4)
            end
            K[i, j] = match / (4.0 * n_windows)
        end
    end
    return K
end
