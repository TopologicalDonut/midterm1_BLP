#= 
Estimation of supply side parameters. 

Estimate the pricing equation for product j in market m: 
    ln(mcⱼₘ) = Xⱼₘθ₃ + ωⱼₘ 

Assume that Xⱼₘ is exogenous: E[ωⱼₘ|Xⱼₘ] = 0

Part 1 - Assume marginal cost pricing: mcⱼₘ = priceⱼₘ
Since Xⱼₘ is exogenous, use OLS to estimate θ₃ coefficeints. 

Part 2 - Use price elasticities and assume firms are in equlibrium.
=#

using CSV               # loading data
using DataFrames        # loading data
using LinearAlgebra     # basic math
using Statistics 
using NLsolve

# Setting up directories for ease of use
tempdir1=@__DIR__
rootdir=tempdir1[1:findfirst("midterm1_BLP",tempdir1)[end]]
cd(joinpath(rootdir, "simulated_data"))

# main datasets
blp_data = CSV.read("merged_w_data.csv", DataFrame)
demand_data = CSV.read("merged_midterm_data.csv", DataFrame) # demand data for price elasticities and simulation of merger
iv_data = CSV.read("iv_supply.csv", DataFrame) # dataframe with all supply instruments
# construct vector of observables Xⱼₘ
X_d = Matrix(demand_data[!, ["price","caffeine_score"]])  # Demand side
X = Matrix(blp_data[!, ["caffeine_score"]]) # exogenous supply variables
# vector of prices pⱼₘ
P = Vector(blp_data[!, "price"])
# observed market shares
S = Vector(blp_data[!, "share"])
# firm and market id numbers
firm_id = Vector(blp_data[!, "firm"])
market_id = Vector(blp_data[!, "marketid"])
# Supply instruments
IV = Matrix(iv_data[!, ["zs1","zs2","zs3","zs4","zs5", "zs6", "zs7"]])
Z = IV

# θ₁ and θ₂ estimates from the demand side
θ₂ = [0.1552762882395271, 0.5888637587155279]
θ₁ = [-1.6106496029235093, 2.044699407331066]

# pre-selected random draws
v_50 = Matrix(CSV.read("50draws.csv", DataFrame, header=0)) # pre-selected random draws from joint normal to simulate 50 individuals
v_5000 = Matrix(CSV.read("5000draws.csv", DataFrame, header=0)) # pre-selected random draws from joint normal to simulate 5000 individuals

# reshape to 3-d arrays: v(market, individual, coefficient draw) 
# the sets of 50 individuals (v_50) is used in most places to estimate market share. 50 is a compromise between speed and precision.
# the sets of 5000 individuals (v_5000) is used for the diagonal of the price elastiticty matrix in supply price elasticities which
# only needs to be calculated once, so greater precision can be achieved. 
v_50 = reshape(v_50, (200,50,2)) # 20 markets, 50 individuals per market, 2 draws per invididual (one for each θ₂ random effect coefficient)
v_5000 = reshape(v_5000, (200,5000,2)) # 20 markets, 5000 individuals per market, 2 draws per invididual (one for each θ₂ random effect coefficient)

#=------------------------------------------------------------------------------------
# Part 2 - Multi-product firms setting prices in equilibrium
# Assume firms set prices simultaneously to maximize static profits across all their products.

Solution to the FOC:
S - Δ(P-MC) = 0
solve for marginal cost: MC = P - Δ⁻¹S

Where Δ is a vector of own and cross price elastiticities, P is a vector of prices
and S is a vector of market shares. 

Finding Δ is the challenging part. See "supply elasiticites.jl" for the calculation
and detailed documentation.

Steps:
- Solve for Δ
- Calculate marginal cost MC
- Use MC to estimate θ₃ with OLS

Note: ~10% of marginal cost estimates turn out to be negative in this dataset. 
They are dropped from analysis before the log transformation. This can be avoided
by estimating supply and demand simultaneously.
=#

# load module with function to calculate price elasticities
cd(joinpath(rootdir, "code"))
include("supply_price_elasticities.jl")
include("demand_functions.jl")

using .supply_price_elasticities
using .demand_functions

# calculate matrix of price elasticities
Δ = price_elasticities(θ₁, θ₂, X_d, S, v_5000, v_50, market_id, firm_id)

# get inverse
Δ⁻¹ = inv(Δ)

# calculate MC
MC = P - Δ⁻¹*S
negative_MC_indices = findall(x -> x < 0, MC) # To make life easier for the merger

## IV Regression 

# drop any negative marginal cost estimates to allow for log transformation
X = X[MC.>0,:]
Z = Z[MC.>0,:]
MC = MC[MC.>0]

# parameter estimates
θ₃ = inv((X'Z)*inv(Z'Z)*(X'Z)') * (X'Z)*inv(Z'Z)*Z'*log.(MC)

# Robust Standard Errors
# residuals
ω = log.(MC) - X*θ₃
# covariance matrix
Σ = Diagonal(ω*ω')
Var_θ₃ = inv(X'X)*(X'*Σ*X)*inv(X'X)
# standard errors
SE_θ₃ = sqrt.(Diagonal(Var_θ₃))

# approximate solution is θ₃ and SE_θ₃.
# θ₃     = [0.006053688185990462]
# SE_θ₃  = [0.0400735852012778]

# OLS
θ₃_OLS = inv(X'X)X'log.(MC) # = 0.006053688185990708

## Checking simulated prices for sanity check 

# Initializing
cd(joinpath(rootdir, "simulated_data"))
ξ = CSV.read("xi.csv", DataFrame)
ξ = Vector(ξ[!,"xi"])

# Need to drop all the rows of negative MC for everything else
X_d = X_d[setdiff(1:size(X_d,1), negative_MC_indices), :]
ξ = ξ[setdiff(1:length(ξ), negative_MC_indices)]
market_id = market_id[setdiff(1:length(market_id), negative_MC_indices)]
firm_id = firm_id[setdiff(1:length(firm_id), negative_MC_indices)]

# There's no guarantee that fixed point methods would converge to price, but it seems to work in this case.
function sim_objective_function(p, θ₁, θ₂, X_d, ξ, v_50, v_5000, market_id, firm_id, MC)
    # Update X_d with the current price vector
    X_d[:, 1] = p

    # Compute shares and elasticities based on the updated x₁
    δ = X_d * θ₁ + ξ
    shares_vector = demand_functions.shares(δ, θ₂, X_d, v_50, market_id)[1]
    elasticities_matrix = price_elasticities(θ₁, θ₂, X_d, shares_vector, v_5000, v_50, market_id, firm_id)

    # Calculate the first order condition's value
    p_new = MC + elasticities_matrix \ shares_vector

    # Return the norm of the FOC values as the objective to minimize
    return p_new
end

p_initial = P[setdiff(1:length(P), negative_MC_indices)]

result = fixedpoint(p -> sim_objective_function(p, θ₁, θ₂, X_d, ξ, v_50, v_5000, market_id, firm_id, MC), p_initial, xtol=1e-6, show_trace=true)
test = result.zero
testmeandiff = mean(p_initial - test) # -0.0048673022682182866 Looks like it worked well.

df_p = DataFrame(Value = test)
df_pmeandiff = DataFrame(Value = testmeandiff)

cd(rootdir)
CSV.write("simulated_data/Simulated_P_original.csv", df_p) 
CSV.write("simulated_data/Simulated_P_original_meandiff.csv", df_pmeandiff)

### Merger Simulation ###
cd(joinpath(rootdir, "simulated_data"))

# main datasets
blp_data = CSV.read("merger_w_data.csv", DataFrame)
X = Matrix(blp_data[!, ["caffeine_score"]]) 
# firm and market id numbers
firm_id = Vector(blp_data[!, "firm"])
market_id = Vector(blp_data[!, "marketid"])

# Need to drop the mising rows of original MC like usual (since we need to use ξ and ω)
X = X[setdiff(1:length(X), negative_MC_indices), :]
firm_id = firm_id[setdiff(1:length(firm_id), negative_MC_indices)]
market_id = market_id[setdiff(1:length(market_id), negative_MC_indices)]
MC_min = exp.(X*θ₃ + ω) # Simulated marginal cost

result = fixedpoint(p -> sim_objective_function(p, θ₁, θ₂, X_d, ξ, v_50, v_5000, market_id, firm_id, MC_min), p_initial, xtol=1e-6, show_trace=true)
p_merger = result.zero
diff = p_merger - p_initial # To compare the vectors
num_negatives = count(x -> x < 0, diff) # 497 negatives, meaning 497 products simulated to have lower prices.

# Convert the vector to a DataFrame
df_p = DataFrame(Value = p_merger)
df_pdiff = DataFrame(Value = diff)
# Write the DataFrame to a CSV file
cd(rootdir)
CSV.write("simulated_data/Simulated_P_minmerger.csv", df_p)
CSV.write("simulated_data/Simulated_P_minmerger_diff.csv", df_pdiff)

## Now doing average caffeine score scenario

cd(joinpath(rootdir, "simulated_data"))

blp_data = CSV.read("merger_w_data2.csv", DataFrame)
X = Matrix(blp_data[!, ["caffeine_score"]]) 
firm_id = Vector(blp_data[!, "firm"])
market_id = Vector(blp_data[!, "marketid"])

X = X[setdiff(1:length(X), negative_MC_indices), :]
firm_id = firm_id[setdiff(1:length(firm_id), negative_MC_indices)]
market_id = market_id[setdiff(1:length(market_id), negative_MC_indices)]
MC_avg = exp.(X*θ₃ + ω) 

result = fixedpoint(p -> sim_objective_function(p, θ₁, θ₂, X_d, ξ, v_50, v_5000, market_id, firm_id, MC_avg), p_initial, xtol=1e-6, show_trace=true)
p_merger = result.zero
diff = p_merger - p_initial # To compare the vectors
num_negatives = count(x -> x < 0, diff) # 496 negatives, meaning 496 products simulated to have lower prices.

# Convert the vector to a DataFrame
df_p = DataFrame(Value = p_merger)
df_pdiff = DataFrame(Value = diff)
# Write the DataFrame to a CSV file
cd(rootdir)
CSV.write("simulated_data/Simulated_P_avgmerger.csv", df_p)
CSV.write("simulated_data/Simulated_P_avgmerger_diff.csv", df_pdiff)

