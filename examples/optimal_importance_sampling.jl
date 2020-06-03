using CrossEntropyMethod
using Distributions, Random
using Plots

base_dist = Categorical([0.74899, 0.2, 0.05, 0.001, 0.00001])
is_dist_0 = Dict{Symbol, Vector{Sampleable}}(:x => [Categorical(5)])

function l(d, s)
    v = s[:x][1]
    -(v in [3,4,5])
end

function w(d, s)
    v = s[:x][1]
    exp(logpdf(base_dist, v) - logpdf(d, s))
end

s = Dict(:x => [5])
w(is_dist_0, s)

is_dist_opt = cross_entropy_method(l, is_dist_0; max_iter = 10, N=1000, weight_fn = w, verbose = true)

is_dist_opt[:x][1].p

