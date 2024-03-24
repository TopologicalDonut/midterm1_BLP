#= BLP instruments =#
# function to enclose the calculation of instruments.
# same code as Demand Side - OLS and 2SLS, packaged as a function to save space.

module demand_instrument_module
export BLP_instruments

#= Two sets of instruments
1. Characteristics of other products from the same company in the same market.
Logic: the characteristics of other products affect the price of a 
given product but not its demand. Alternatively, firms decide product characteristics X 
before observing demand shocks ξ. 
2. Characteristics of other products from different companies in the same market.
Logic: the characteristics of competing products affects the price of a
given product but not its demand. Alternatively, other firms decide their product
characteristics X without observing the demand shock for the given product ξ.
=#

# Note that for the midterm there are no instruments for other products of the same company.
function BLP_instruments(X, id, cdid, firmid)

n_products = size(id,1) # number of observations = 600

# initialize arrays to hold instruments. 
IV_rivals = zeros(n_products,1)

# loop through every product in every market (every observation)
for j in 1:n_products
    # 2. Set of instruments from rival product characteristics
    # get index of all products from different firms (firmid) in the same market/year (cdid)
    rival_index = (firmid.!=firmid[j]) .* (cdid.==cdid[j])
    # x variable values for other products (excluding price)
    rival_x_values = X[rival_index,:]
    # sum along columns
    IV_rivals[j,:] = sum(rival_x_values, dims=1)
end

# vector of observations and instruments
IV = [X IV_rivals]

return IV
end

end # end module 
