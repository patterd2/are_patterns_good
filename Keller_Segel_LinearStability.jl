# Linear Stability Analysis of Keller-Segel Model with logistic growth

using DifferentialEquations
using Plots, Measures, LaTeXStrings

# Parameters
Dn = 0.1
Dc = 0.1
χ = 0.41
α = 0.5
β = 1.0
K = 2.0

L = 10.0              # Length of the domain
M = 200             # Number of modes

# trace always negative, just check the determinant
D(k,χ,β) = (1+Dn*((π*k/L)^2))*(β+Dc*((π*k/L)^2)) - α*χ*K*((π*k/L)^2)

# Define the range of x values
k = 0:1:M

# Plot the function
plt = scatter(k, D.(k,χ,β), markersize = 3, 
    label="D(k)", title="Plot of det(J)", xlabel="k",
    xlims=(0,20), ylims=(-1.5,1.5))
hline!([0], label="", linestyle=:dash, color=:black)
annotate!(10, 1.35, text("χ = $(χ)", :black, 12))
display(plt)

# Define the list of unstable wave numbers
unstable_wave_numbers = D.(0:1:M,χ,β) .< 0

# Print the unstable wave numbers for fixed parameters
println("Unstable wave numbers: ", findall(!iszero, unstable_wave_numbers))

# 2D plot of instability regions
χ_range = LinRange(0, 1.2, 800)
β_range = LinRange(0, 5, 2000)

# Define function
f(x1, y1) = y1 - ((y1*Dn + Dc - α*x1*K)^2)/(4*Dn*Dc)

# Compute grid values
z = [f(xi, yi) for xi in χ_range, yi in β_range]  # note: y rows, x columns

# Plot heatmap
# Define French flag gradient: blue → white → red
french_flag = cgrad([RGB(1,1,1), RGB(1,0,0)], 256)

heatmap(χ_range, β_range, -sign.(transpose(z)), c=french_flag, 
    xlabel=latexstring("χ"), ylabel=latexstring("β"), colorbar=false, yguidefontrotation=-90,
    xguidefontsize=20, yguidefontsize=20, left_margin=10mm, bottom_margin=2mm)
annotate!(0.32, 2.7, text("Homogeneous \nsolution stable", :black, 12))
hline!([1], label="", linestyle=:dash, linewidth = 2, color=:black)
#vline!([0.403], label="", linestyle=:dash, color=:red)

