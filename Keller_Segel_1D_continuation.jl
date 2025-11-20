using Revise
using SparseArrays, DiffEqOperators, Parameters
import LinearAlgebra: I, norm
using BifurcationKit
using Plots, Measures, LaTeXStrings
const BK = BifurcationKit

# Discretization
Nx = 400
L = 10.0
X = LinRange(0, L, Nx)
dx = L / (Nx - 1)

# Parameters associated with the Keller-Segel model
parKS = (Dn = 0.1, Dc = 0.1, χ = 0.165, α = 0.5, β = 1.0, K = 5.0)

# Define the functional for the Keller-Segel model
function R_KS(u, par)
    @unpack Dn, Dc, χ, α, β, K = par
    n = @view u[1:Nx]
    c = @view u[Nx+1:2Nx]
    out = similar(u)

    dn = @view out[1:Nx]
    dc = @view out[Nx+1:2Nx]

    # Pre-allocate arrays for derivatives
    ∂²n∂x² = similar(n)
    ∂²c∂x² = similar(c)
    ∂c∂x = similar(c)
    ∂n∂c∂x = similar(n)

    # Compute derivatives
    @inbounds begin
        # Second derivatives
        ∂²n∂x²[2:end-1] .= (n[3:end] .- 2n[2:end-1] .+ n[1:end-2]) / dx^2
        ∂²c∂x²[2:end-1] .= (c[3:end] .- 2c[2:end-1] .+ c[1:end-2]) / dx^2

        ∂²n∂x²[1] = (n[2] - n[1]) / dx^2
        ∂²n∂x²[end] = (n[end-1] - n[end]) / dx^2

        ∂²c∂x²[1] = (c[2] - c[1]) / dx^2
        ∂²c∂x²[end] = (c[end-1] - c[end]) / dx^2

        # First derivative of c
        ∂c∂x[2:end-1] .= (c[3:end] .- c[1:end-2]) / (2dx)
        ∂c∂x[1] = 0.0
        ∂c∂x[end] = 0.0

        # Aggregation term
        ∂n∂c∂x[2:end-1] .= ((n[3:end] .* ∂c∂x[3:end]) .- (n[1:end-2] .* ∂c∂x[1:end-2])) / (2dx)
        ∂n∂c∂x[1] = 0.0
        ∂n∂c∂x[end] = 0.0
    end

    # Update equations
    @. dn = Dn * ∂²n∂x² - χ * ∂n∂c∂x + n * (1 - n / K)
    @. dc = Dc * ∂²c∂x² + α * n - β * c

    return out
end

# starting solutions
sol0 = [fill(parKS.K, Nx); fill(parKS.K*parKS.α/parKS.β, Nx)]

# bring in solution from the timestepper, can refine further
sol_pattern = [ sol.u[2][1:Nx]; sol.u[2][Nx+1:2Nx]]

# Bifurcation Problem
prob = BifurcationProblem(R_KS, sol_pattern, parKS, (@lens _.χ);
    record_from_solution = (x, p) -> (n2 = dx*sum(x[1:Nx]), s = sum(x)),
    plot_solution = (x, p; kwargs...) -> (plot!(X[1:Nx], x[1:Nx]; ylabel="n(x)", label="x", kwargs...))
)

# newton corrections of the equilibrium
#optnewton = NewtonPar(verbose = true, tol = 1e-8, max_iterations = 100)
#sol_refined = @time newton(prob, optnewton)

# bifurcation problem with patterned start
prob2 = BifurcationProblem(R_KS, sol0, parKS, (@lens _.χ);
    record_from_solution = (x, p) -> (n2 = dx*sum(x[1:Nx]), s = sum(x)),
    plot_solution = (x, p; kwargs...) -> (plot!(X[1:Nx], x[1:Nx]; ylabel="n(x)", label="x", kwargs...))
)

# Continuation parameters and settings
opts = ContinuationPar(
    dsmin = 0.0000001,
    dsmax = 0.005,
    ds = 0.001,
    p_min = 0.01,    
    p_max = 0.25,
    newton_options = NewtonPar(max_iterations = 50, tol = 1e-12),
    max_steps = 500,
    plot_every_step = 1
)

opts2 = ContinuationPar(
    dsmin = 0.0000001,
    dsmax = 0.01,
    ds = -0.01,
    p_min = 0.01,    
    p_max = 0.25,
    newton_options = NewtonPar(max_iterations = 50, tol = 1e-12),
    max_steps = 500,
    plot_every_step = 1
)

# continuation of equilibria
# br = continuation(
# 	re_make(prob, u0 = sol_refined.u), PALC(), opts2;
# 	plot = true, verbosity = 1,
# 	normC = norminf)

br = continuation(prob, PALC(), opts2;
    plot = false, verbosity = 1,
    normC = norminf, bothside = true)

br2 = continuation(prob2, PALC(), opts;
    plot = false, verbosity = 1,
    normC = norminf, bothside = true)

br3 = continuation(br2, 2, setproperties(opts; ds = 0.001, detect_bifurcation = 3, plot_every_step = 5, max_steps = 170);  nev = 30,
	plot = true, verbosity = 0,
	normC = norminf)



# Plot the bifurcation diagram with manual branch switching
col = [stb ? :red : :black for stb in br.branch.stable]
branch_style = [stb ? :solid : :solid for stb in br.branch.stable]
plot(br, plotfold=true, linewidth = 2, linestyle = branch_style, 
    markersize = 3, color=col, plotspecialpoints=true)

col = [stb ? :red : :black for stb in br2.branch.stable]
branch_style = [stb ? :solid : :solid for stb in br2.branch.stable]
plot!(br2, plotfold=true, linewidth = 2, linestyle = branch_style, 
    markersize = 3, color=col, plotspecialpoints=true)

col = [stb ? :red : :black for stb in br3.branch.stable]
branch_style = [stb ? :solid : :solid for stb in br3.branch.stable]
plot!(br3, color=col, linestyle = branch_style, plotspecialpoints=false, plotfold = true, linewidthstable = 3, linewidthunstable = 3)

#col2 = [stb ? :red : :black for stb in br_pattern.branch.stable]
#branch_style2 = [stb ? :dash : :dash for stb in br_pattern.branch.stable]


#plot!(br_pattern, plotfold=true, linewidth = 2, linestyle = branch_style2, 
#    markersize = 2, color=col2, plotspecialpoints=true)

    #plot!(br3, plotfold=true, linewidthstable = 3, markersize = 2,linestyle=:dot)
#plot!(br4, plotfold=true, linewidthstable = 3, markersize = 2)
#plot!(br_pattern, plotfold=true, linewidthstable = 3, markersize = 2)
plot!(ylabel=latexstring("‖n~‖_{L_1}"), 
    xlabel=latexstring("χ"),
    yguidefontrotation=-90,
    left_margin=14mm,
    bottom_margin=2mm,
    xguidefontsize=20,
    yguidefontsize=20,
    xlim=(0.15, 0.165), 
    ylim=(48, 50.5))



# Save the bifurcation diagram
#savePlot = plot(diagram; plotfold=true, linewidthstable=4, markersize=1, putspecialptlegend=false)
#savefig("plots/Keller_Segel_bifurcationZoom_May8.svg")