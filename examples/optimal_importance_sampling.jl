using CrossEntropyMethod
using Distributions, Random
using Plots
# The goal of this example is to show how to estimate the optimal importance sampling distribution

# Actual distibution
base_dist = Categorical([0.74899, 0.2, 0.05, 0.001, 0.00001])

# starting sampling distribution
is_dist_0 = Dict{Symbol, Vector{Sampleable}}(:x => [Categorical(5)])

# Sample a random time series from the starting distribution
rand(is_dist_0)

# Define a loss function that identifies the values of 3,4,5 as those of intereste
function l(d, s)
    v = s[:x][1]
    -(v in [3,4,5])
end

# Define the likelihood ratio weighting function
function w(d, s)
    v = s[:x][1]
    exp(logpdf(base_dist, v) - logpdf(d, s))
end

# show what the weighting function does
s = Dict(:x => [5])
w(is_dist_0, s)

# Run the optimization
is_dist_opt = cross_entropy_method(l, is_dist_0; max_iter = 4, N=10000, weight_fn = w, verbose = true)

# See the final distribution.
is_dist_opt[:x][1].p

