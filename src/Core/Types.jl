"""
    Types.jl

Defines the core data structures for AdvancedGenomics.jl.
Designed for high performance and extensibility.
"""

abstract type AbstractGenotypeMatrix end

"""
    GenotypeMatrix{T} <: AbstractGenotypeMatrix

A wrapper around a matrix of genotype data.
Can hold dense (Matrix{T}) or sparse (SparseMatrixCSC{T}) data.
T is typically Int8 (0, 1, 2) or Float32 (imputed dosages).
"""
struct GenotypeMatrix{T <: Real, M <: AbstractMatrix{T}} <: AbstractGenotypeMatrix
    data::M
    individuals::Vector{String}
    snps::Vector{String}
    
    function GenotypeMatrix(data::M, individuals::Vector{String}, snps::Vector{String}) where {T, M <: AbstractMatrix{T}}
        @assert size(data, 1) == length(individuals) "Number of rows in data must match number of individuals"
        @assert size(data, 2) == length(snps) "Number of columns in data must match number of SNPs"
        new{T, M}(data, individuals, snps)
    end
end

"""
    PhenotypeData

Stores phenotype values and covariates.
"""
struct PhenotypeData
    data::DataFrame
    id_col::Symbol
    trait_cols::Vector{Symbol}
    covariate_cols::Vector{Symbol}
end

"""
    MultiOmicsData

Stores multi-omics data (e.g., Gene Expression, Methylation).
"""
struct MultiOmicsData{T <: Real}
    data::Matrix{T}
    individuals::Vector{String}
    features::Vector{String}
    omics_type::String # e.g., "Transcriptomics", "Methylomics"
end

"""
    Pedigree

Stores pedigree information for constructing A-matrix.
"""
struct Pedigree
    id::Vector{String}
    sire::Vector{Union{String, Missing}}
    dam::Vector{Union{String, Missing}}
end
