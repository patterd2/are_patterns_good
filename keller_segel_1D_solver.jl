# Keller-Segel model with logistic growth 
using DifferentialEquations
using Plots

Tmax = 500.0          # Maximum time

# Parameters
Dn = 0.1
Dc = 0.1
χ = 0.5
α = 0.5
β = 1.0
K = 2.0

# Spatial and temporal grid
L = 10.0              # Length of the domain
Nx = 400             # Number of spatial points
dx = L / (Nx - 1)
x = LinRange(0, L, Nx) # Spatial grid

# Initial conditions
n0(x) = exp(-((x - L/2)^2) / 2.0)
c0(x) = 0.5 * exp(-((x - L/2)^2) / 2.0)

# PDE system with aggregation term
function pde_system!(du, u, p, t)
    N = Nx
    n = @view u[1:N]    # First N points correspond to n
    c = @view u[N+1:2N] # Last N points correspond to c
    
    dn = @view du[1:N]
    dc = @view du[N+1:2N]

    # Pre-allocate arrays for derivatives (reuse memory)
    ∂²n∂x² = similar(n)
    ∂²c∂x² = similar(c)
    ∂c∂x = similar(c)
    ∂n∂c∂x = similar(n)

    # Compute derivatives and update equations
    @inbounds begin
        # Compute second derivatives (finite differences with no-flux BCs)
        ∂²n∂x²[2:end-1] .= (n[3:end] .- 2n[2:end-1] .+ n[1:end-2]) / dx^2
        ∂²c∂x²[2:end-1] .= (c[3:end] .- 2c[2:end-1] .+ c[1:end-2]) / dx^2

        # Boundary conditions for second derivatives
        ∂²n∂x²[1] = (n[2] - n[1]) / dx^2
        ∂²n∂x²[end] = (n[end-1] - n[end]) / dx^2

        ∂²c∂x²[1] = (c[2] - c[1]) / dx^2
        ∂²c∂x²[end] = (c[end-1] - c[end]) / dx^2

        # Compute first derivative of c (finite differences with no-flux BCs)
        ∂c∂x[2:end-1] .= (c[3:end] .- c[1:end-2]) / (2dx)
        ∂c∂x[1] = 0.0
        ∂c∂x[end] = 0.0

        # Compute the derivative of n * ∂c/∂x (aggregation term)
        ∂n∂c∂x[2:end-1] .= ((n[3:end] .* ∂c∂x[3:end]) .- (n[1:end-2] .* ∂c∂x[1:end-2])) / (2dx)
        ∂n∂c∂x[1] = 0.0
        ∂n∂c∂x[end] = 0.0
    end

    # Update equations
    @. dn = Dn * ∂²n∂x² - χ * ∂n∂c∂x + n * (1 - n / K)
    @. dc = Dc * ∂²c∂x² + α * n - β * c
end

# Initial conditions vector

# u0 = [n0.(x); c0.(x)]  # Combine initial conditions for n and c

u0 = [sol.u[2][1:Nx]; sol.u[2][Nx+1:2Nx]] # continue from previous solution

# Time span
tspan = (0.0, Tmax)

# Solve the PDE system with reduced memory usage
@time begin
    prob = ODEProblem(pde_system!, u0, tspan)
    sol = solve(prob, Tsit5(), reltol=1e-6, abstol=1e-6, dtmax=1e-2, save_everystep=false, verbose=false)
end

# Plot results (only the final state is available)
plot(x, sol.u[2][1:Nx], label=latexstring("n(x, t = 500)"), xlabel="x")
plot!(x, sol.u[2][Nx+1:2Nx], label=latexstring("c(x, t = 500)"), xlabel="x")
