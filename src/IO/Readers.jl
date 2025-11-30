"""
    Readers.jl

Functions for reading genomic data formats.
"""

using CSV
using DataFrames

"""
    read_phenotypes(file::String; id_col::Symbol=:ID, trait_cols::Vector{Symbol}=[], covariate_cols::Vector{Symbol}=[])

Reads a CSV file containing phenotype and covariate data.
"""
function read_phenotypes(file::String; id_col::Symbol=:ID, trait_cols::Vector{Symbol}=Symbol[], covariate_cols::Vector{Symbol}=Symbol[])
    df = CSV.read(file, DataFrame)
    
    # Validation
    if !(id_col in propertynames(df))
        error("ID column $id_col not found in file.")
    end
    
    # Auto-detect traits if not provided (assuming all numeric non-ID/non-covariate cols are traits)
    if isempty(trait_cols)
        exclude = [id_col; covariate_cols]
        trait_cols = [n for n in propertynames(df) if eltype(df[!, n]) <: Number && !(n in exclude)]
    end
    
    return PhenotypeData(df, id_col, trait_cols, covariate_cols)
end

"""
    read_genotypes(file::String; format::String="csv")

Reads genotype data. Currently supports simple CSV format (Rows=Ind, Cols=SNPs).
Future extensions: PLINK binary (.bed), VCF.
"""
function read_genotypes(file::String; format::String="csv")
    if format == "csv"
        df = CSV.read(file, DataFrame)
        # Assuming first column is ID
        ids = map(String, df[!, 1])
        snps = names(df)[2:end]
        data = Matrix{Float32}(df[!, 2:end])
        return GenotypeMatrix(data, ids, snps)
    else
        error("Format $format not supported yet.")
    end
end

"""
    read_omics(file::String, type::String)

Reads multi-omics data matrix.
"""
function read_omics(file::String, type::String)
    df = CSV.read(file, DataFrame)
    ids = string.(df[!, 1])
    features = names(df)[2:end]
    data = Matrix{Float32}(df[!, 2:end])
    return MultiOmicsData(data, ids, features, type)
end
