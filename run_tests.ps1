$juliaPath = "C:\Users\MeiBujun\.julia\juliaup\julia-1.12.2+0.x64.w64.mingw32\bin\julia.exe"
Write-Host "Running tests using Julia at: $juliaPath"
& $juliaPath --project=. test/runtests.jl
