"""
    MME.jl

Mixed Model Equations (MME) module.
Constructs and solves:
[X'X    X'Z] [b]   [X'y]
[Z'X    Z'Z+I*lambda] [u] = [Z'y]
"""

using LinearAlgebra
using SparseArrays
using Statistics

"""
    build_MME(y, X, Z, K_inv, lambda)

Constructs LHS and RHS of the Mixed Model Equations.
y: Phenotypes
X: Fixed effect design matrix
Z: Random effect design matrix
K_inv: Inverse of relationship matrix (A^-1 or G^-1)
lambda: Variance ratio (sigma_e2 / sigma_g2)
"""
function build_MME(y::Vector{Float64}, X::AbstractMatrix, Z::AbstractMatrix, K_inv::AbstractMatrix, lambda::Float64)
    # Ensure dimensions match
    n = length(y)
    p = size(X, 2)
    q = size(Z, 2)
    
    # LHS blocks
    XtX = X' * X
    XtZ = X' * Z
    ZtX = Z' * X
    ZtZ = Z' * Z
    
    # Add regularization to ZtZ
    # ZtZ + K_inv * lambda
    # Handle sparse or dense K_inv
    if issparse(K_inv) || issparse(ZtZ)
        lower_right = ZtZ + K_inv .* lambda
        
        # Construct sparse LHS
        # [XtX  XtZ]
        # [ZtX  lower_right]
        
        # Block construction for sparse is tricky in Julia if types differ
        # Convert all to sparse
        LHS = [sparse(XtX) sparse(XtZ);
               sparse(ZtX) lower_right]
    else
        lower_right = ZtZ + K_inv .* lambda
        LHS = [XtX XtZ;
               ZtX lower_right]
    end
    
    # RHS
    Xty = X' * y
    Zty = Z' * y
    RHS = [Xty; Zty]
    
    return (LHS=LHS, RHS=RHS)
end

"""
    solve_MME(LHS, RHS)

Solves the MME system.
Returns (beta, u).
"""
function solve_MME(LHS, RHS, p::Int)
    # p is number of fixed effects to split the solution
    
    # Check if sparse
    if issparse(LHS)
        # Use sparse solver
        sol = LHS \ RHS
    else
        sol = LHS \ RHS
    end
    
    beta = sol[1:p]
    u = sol[p+1:end]
    
    return (beta=beta, u=u)
end
