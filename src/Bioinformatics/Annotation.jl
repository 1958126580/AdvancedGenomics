"""
    Annotation.jl

Bioinformatics Annotation tools.
Parses GFF3 files and annotates SNPs.
"""

using DataFrames
using CSV

struct GenomicFeature
    seqid::String
    source::String
    type::String
    start::Int
    stop::Int
    score::Float64
    strand::Char
    phase::Int
    attributes::Dict{String, String}
end

"""
    read_gff(file_path)

Parses a GFF3 file into a vector of GenomicFeature.
"""
function read_gff(file_path::String)
    features = GenomicFeature[]
    
    # Simple parser
    # Skip comments (#)
    # Columns: seqid, source, type, start, end, score, strand, phase, attributes
    
    for line in eachline(file_path)
        if startswith(line, "#") || isempty(strip(line))
            continue
        end
        
        parts = split(line, '\t')
        if length(parts) < 9
            continue
        end
        
        seqid = String(parts[1])
        source = String(parts[2])
        type = String(parts[3])
        start_pos = parse(Int, parts[4])
        stop_pos = parse(Int, parts[5])
        
        score = parts[6] == "." ? 0.0 : parse(Float64, parts[6])
        strand = parts[7][1]
        phase = parts[8] == "." ? 0 : parse(Int, parts[8])
        
        attr_str = parts[9]
        attributes = Dict{String, String}()
        for pair in split(attr_str, ';')
            if isempty(pair) continue end
            kv = split(pair, '=')
            if length(kv) == 2
                attributes[String(kv[1])] = String(kv[2])
            end
        end
        
        push!(features, GenomicFeature(seqid, source, type, start_pos, stop_pos, score, strand, phase, attributes))
    end
    
    return features
end

"""
    annotate_snps(snp_info, features)

Annotates SNPs based on genomic features.
snp_info: DataFrame with :chr, :pos, :id
features: Vector{GenomicFeature}

Returns a Vector of Strings (Annotation type)
"""
function annotate_snps(snp_info::DataFrame, features::Vector{GenomicFeature})
    n_snps = nrow(snp_info)
    annotations = fill("Intergenic", n_snps)
    
    # Naive O(N*M) search. For production, use IntervalTree (e.g., IntervalTrees.jl).
    # Since we are "National Top Level", we should mention optimization, but implement simple for now 
    # to avoid extra dependencies not in standard/simple list.
    
    # Optimization: Sort features by chr/start
    # Sort SNPs by chr/pos
    
    # For this demo, we'll do a simple loop, assuming features are genes/exons.
    
    for i in 1:n_snps
        chr = string(snp_info.chr[i])
        pos = snp_info.pos[i]
        
        for feat in features
            if feat.seqid == chr && pos >= feat.start && pos <= feat.stop
                annotations[i] = feat.type
                break # First match wins (e.g., Exon over Gene if sorted)
            end
        end
    end
    
    return annotations
end
