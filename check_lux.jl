using Lux
println("Lux loaded")
try
    println(MultiHeadAttention)
    println("MultiHeadAttention found")
catch e
    println("MultiHeadAttention NOT found")
    println(e)
end
