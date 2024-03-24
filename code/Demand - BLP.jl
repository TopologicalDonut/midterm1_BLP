### Midterm 1 Computational Econ ###
### Code modified by Stephen Min ###

# Setting up directories for ease of use 
tempdir1=@__DIR__
rootdir=tempdir1[1:findfirst("midterm1_BLP",tempdir1)[end]]
# Load key functions and packages -------------------------------------------------
cd(joinpath(rootdir, "code"))

include("demand_functions.jl")    # module with custom BLP functions (objective function and σ())
include("demand_instruments.jl")  # module to calculate BLP instruments
include("demand_derivatives.jl")  # module with gradient function 

using .demand_functions
using .demand_instrument_module
using .demand_derivatives

using CSV               # loading data
using DataFrames        # loading data
using LinearAlgebra     # basic math
using Statistics        # for mean


# Load key data ------------------------------------------------------------------
cd(joinpath(rootdir, "simulated_data"))

blp_data = CSV.read("merged_midterm_data.csv", DataFrame) # dataframe with all observables 
iv_data = CSV.read("iv_demand.csv", DataFrame) # dataframe with all instruments
v_50 = Matrix(CSV.read("50draws.csv", DataFrame, header=0)) # pre-selected random draws from joint normal to simulate 50 individuals
# reshape to 3-d arrays: v(market, individual, coefficient draw) 
v_50 = reshape(v_50, (200,50,2)) # 200 markets, 50 individuals per market, 2 draws per invididual (one for each θ₂ random effect coefficient)

# Load X variables. 600x2 matrices respectively
X = Matrix(blp_data[!, ["price","caffeine_score"]]) # exogenous x variables and price
# Load Y variable market share. 600x1 vector
share = Vector(blp_data[!,"share"])
# Load IV variables. 600x7 matrix
IV = Matrix(iv_data[!, ["zd1","zd2","zd3","zd4","zd5", "zd6", "zd7"]])
# product, market, and firm ids 
id = Vector(blp_data[!,"id"])
cdid = Vector(blp_data[!,"marketid"])
firmid = Vector(blp_data[!,"firm"])

# BLP instruments. Price (column 1) not included in BLP instruments.
Z = IV
Z_gen = BLP_instruments(X[:,Not(1)], id, cdid, firmid)

# Minimize objective function -----------------------------------------------------
using Optim             # for minimization functions
using BenchmarkTools    # for timing/benchmarking functions

# θ₂ guess values. Initialze elements as floats.
θ₂ = [0.5, 0.5] 

# test run and timing of objective function and gradient
# Q, θ₁, ξ, 𝒯 = demand_objective_function(θ₂,X,share,Z,v_50,cdid) # returns 4 values  
# @btime demand_objective_function($θ₂,$X,$share,$Z,$v_50,$cdid)  
# Usually <100ms. Speed varies depending on θ₂.

# g = gradient(θ₂,X,Z,v_50,cdid,ξ,𝒯)
# @btime gradient($θ₂,$X,$Z,$v_50,$cdid,$ξ,$𝒯)
# ~ 1.1 seconds. 


# temporary ananomyous functions for objective function and gradient
function f(θ₂)
    # run objective function and get key outputs
    Q, θ₁, ξ, 𝒯 = demand_objective_function(θ₂,X,share,Z,v_50,cdid)
    # return objective function value
    return Q
end

function ∇(storage, θ₂)
    # run objective function to update ξ and 𝒯 values for new θ₂
    Q, θ₁, ξ, 𝒯 = demand_objective_function(θ₂,X,share,Z,v_50,cdid)
    # calculate gradient and record value
    g = gradient(θ₂,X,Z,v_50,cdid,ξ,𝒯)
    storage[1] = g[1]
    storage[2] = g[2]
    storage[3] = g[3]
    storage[4] = g[4]
    storage[5] = g[5]
end

# optimization routines
result = optimize(f, θ₂, NelderMead(), Optim.Options(x_tol=1e-6, iterations=500, show_trace=true, show_every=10))

# get results 
θ₂ = Optim.minimizer(result)
θ₁ = demand_objective_function(θ₂,X,share,Z,v_50,cdid)[2]
ξ = demand_objective_function(θ₂,X,share,Z,v_50,cdid)[3]
df = DataFrame(xi = ξ)
CSV.write("xi.csv", df)
𝒯 = demand_objective_function(θ₂,X,share,Z,v_50,cdid)[4]

# solution
# θ₂ = [0.1552762882395271, 0.5888637587155279]
# θ₁ = [-1.6106496029235093, 2.044699407331066]

## Generated IV Estimates
function f(θ₂)
    # run objective function and get key outputs
    Q, θ₁, ξ, 𝒯 = demand_objective_function(θ₂,X,share,Z_gen,v_50,cdid)
    # return objective function value
    return Q
end

result = optimize(f, θ₂, NelderMead(), Optim.Options(x_tol=1e-6, iterations=500, show_trace=true, show_every=10))
θ₂_gen = Optim.minimizer(result)
θ₁_gen = demand_objective_function(θ₂,X,share,Z_gen,v_50,cdid)[2]

# solution
# θ₂_gen = [0.2579144323592907, 0.5888637587155279]
# θ₁_gen = [ -1.6062457464694542, 2.031710007367826]