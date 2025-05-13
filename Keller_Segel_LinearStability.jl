# Linear Stability Analysis of Keller-Segel Model with logistic growth

using DifferentialEquations
using Plots

# Parameters
Dn = 0.1
Dc = 0.1
χ = 0.4
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
unstable_wave_numbers = D.(0:1:M) .< 0

# Print the unstable wave numbers for fixed parameters
println("Unstable wave numbers: ", findall(!iszero, unstable_wave_numbers))

# 2D plot of instability regions
# Create a grid of parameter values
