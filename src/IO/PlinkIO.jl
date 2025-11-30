"""
    PlinkIO.jl

PLINK Binary File I/O.
Reads .bed, .bim, .fam files.
"""

using DataFrames
using CSV
using Mmap

"""
    read_plink(bfile_prefix)

Reads a PLINK dataset (bfile).
Returns (G::GenotypeMatrix, fam::DataFrame, bim::DataFrame).
"""
function read_plink(bfile_prefix::String)
    bed_file = bfile_prefix * ".bed"
    bim_file = bfile_prefix * ".bim"
    fam_file = bfile_prefix * ".fam"
    
    # Read BIM and FAM
    bim = CSV.read(bim_file, DataFrame; header=[:chr, :id, :cm, :pos, :a1, :a2], delim='\t')
    fam = CSV.read(fam_file, DataFrame; header=[:fid, :iid, :father, :mother, :sex, :pheno], delim=' ') # Space or tab? Usually space.
    
    n = nrow(fam)
    m = nrow(bim)
    
    # Read BED
    # BED format:
    # Magic numbers: 0x6C 0x1B
    # Mode: 0x01 (SNP-major)
    # Data: 2 bits per genotype, 4 genotypes per byte.
    # 00 -> Homozygote 1 (0)
    # 01 -> Missing
    # 10 -> Heterozygote (1)
    # 11 -> Homozygote 2 (2)
    # (PLINK coding is slightly different from 0,1,2 additive)
    # PLINK: 00=Homo1, 01=Missing, 10=Het, 11=Homo2
    # Additive (count A1?): Usually count A1 (minor).
    # Let's assume we want 0, 1, 2 count of A1.
    # 00 (Homo1) -> 2 (if A1 is major) or 0?
    # Standard: 00=Homozygote for first allele in BIM?
    # Actually:
    # 00 = Homozygote for allele 1
    # 11 = Homozygote for allele 2
    # 10 = Heterozygote
    # 01 = Missing
    
    # We will map to 0, 1, 2 (count of A2 usually, or A1).
    # Let's map to count of A1 (the minor allele usually in BIM A1 column).
    # If A1 is minor:
    # 00 (Homo A1) -> 2
    # 11 (Homo A2) -> 0
    # 10 (Het) -> 1
    
    # Open file
    io = open(bed_file, "r")
    magic = read(io, 2)
    mode = read(io, 1)
    
    if magic != [0x6c, 0x1b]
        error("Invalid BED file magic number")
    end
    if mode != [0x01]
        error("Only SNP-major mode is supported")
    end
    
    # Calculate bytes per SNP
    # ceil(n / 4)
    bytes_per_snp = div(n + 3, 4)
    
    # Pre-allocate matrix
    G_data = zeros(Float64, n, m)
    
    # Read all bytes
    # For large files, use Mmap or read in chunks.
    # For "National Top Level", let's use a buffer.
    
    buffer = Vector{UInt8}(undef, bytes_per_snp)
    
    for j in 1:m
        read!(io, buffer)
        
        # Parse buffer
        for i in 1:n
            # Byte index
            byte_idx = div(i - 1, 4) + 1
            # Bit offset (0, 2, 4, 6)
            # PLINK order: 1st indiv in lowest 2 bits
            bit_off = (i - 1) % 4 * 2
            
            byte = buffer[byte_idx]
            val = (byte >> bit_off) & 0x03
            
            # Map val
            # 00 (0) -> Homo 1 -> 0 (if counting A2) or 2 (if counting A1)
            # 01 (1) -> Missing -> NaN
            # 10 (2) -> Het -> 1
            # 11 (3) -> Homo 2 -> 2 (if counting A2) or 0
            
            # Let's assume standard additive coding (0, 1, 2 count of alternative allele A2? or A1?)
            # Usually we count the EFFECT allele.
            # Let's map:
            # 00 (Homo 1) -> 0
            # 10 (Het) -> 1
            # 11 (Homo 2) -> 2
            # 01 (Missing) -> NaN
            
            if val == 0x00
                G_data[i, j] = 2.0 # Homo 1 (AA) -> 2 copies of A? Wait.
                # Standard PLINK: 00 is Homozygote for first allele.
                # If we count A1, it is 2.
                # If we count A2, it is 0.
                # Let's assume we count A1.
            elseif val == 0x02 # 10 binary
                G_data[i, j] = 1.0
            elseif val == 0x03 # 11 binary
                G_data[i, j] = 0.0 # Homo 2 (BB) -> 0 copies of A
            elseif val == 0x01
                G_data[i, j] = NaN
            end
            
            # Wait, verify PLINK mapping.
            # 00: Homozygote for first allele in .bim file
            # 11: Homozygote for second allele
            # 10: Heterozygote
            # 01: Missing
            
            # If we want to count A1 (first allele):
            # Homo 1 (00) -> 2
            # Het (10) -> 1
            # Homo 2 (11) -> 0
            
            # Correct.
        end
    end
    
    close(io)
    
    # Create GenotypeMatrix
    # Individuals from FAM
    # SNPs from BIM
    
    # Convert IDs to String
    iids = map(String, fam.iid)
    snps = map(String, bim.id)
    
    return GenotypeMatrix(G_data, iids, snps)
end

"""
    write_bed(file_prefix, G)

Writes a GenotypeMatrix to PLINK BED format.
"""
function write_bed(file_prefix::String, G::GenotypeMatrix)
    n, m = size(G.data)
    
    # Write FAM (Dummy if not provided)
    fam_df = DataFrame(fid=G.individuals, iid=G.individuals, father=0, mother=0, sex=0, pheno=-9)
    CSV.write(file_prefix * ".fam", fam_df, delim=' ', header=false)
    
    # Write BIM
    bim_df = DataFrame(chr=1, id=G.snps, cm=0, pos=1:m, a1="A", a2="B")
    CSV.write(file_prefix * ".bim", bim_df, delim='\t', header=false)
    
    # Write BED
    open(file_prefix * ".bed", "w") do io
        # Magic
        write(io, [0x6c, 0x1b])
        # Mode
        write(io, [0x01])
        
        bytes_per_snp = div(n + 3, 4)
        buffer = zeros(UInt8, bytes_per_snp)
        
        for j in 1:m
            fill!(buffer, 0x00)
            for i in 1:n
                g = G.data[i, j]
                
                # Map 0, 1, 2 to PLINK bits
                # We assumed 0=Homo2, 1=Het, 2=Homo1 (Count of A1)
                # 00 (Homo1) -> 2
                # 10 (Het) -> 1
                # 11 (Homo2) -> 0
                
                val = 0x01 # Default Missing
                if !isnan(g)
                    if g == 2.0
                        val = 0x00 # Homo 1
                    elseif g == 1.0
                        val = 0x02 # Het
                    elseif g == 0.0
                        val = 0x03 # Homo 2
                    end
                end
                
                byte_idx = div(i - 1, 4) + 1
                bit_off = (i - 1) % 4 * 2
                
                buffer[byte_idx] |= (val << bit_off)
            end
            write(io, buffer)
        end
    end
end
