# CrossEntropyMethod.jl
This package provides an implementation of the cross entropy method for optimizing multivariate time series distributions.
Suppose we have a timeseries `X = {x1, ..., xN}` where each `xi` is a vector of dimension `m`. This package provides optimization for two different scenarios:
1. The time series is sampled IID from a single distribution `p`: `xi ~ p(x)`. In this case, the distribution is represented as a `Dict{Symbol, Tuple{Sampleable, Int64}}`. The dictionary will contain `m` symbols, one for each variable in the series. The `Sampleable` object represents `p` and the integer is the length of the timeseries (`N`)
2. The time series is sampled from a different distribution at each timestep `pi`: `x_i ~ p_i(x)`. In this case, the distribution is represented as a `Dict{Symbol, Tuple{Sampleable, Int64}}`

Note: The `Sampleable` objects must support the `Distributions.jl` function `logpdf` and `fit`.

## Usage
See the `examples/` folder for an example use case
The main function is `cross_entropy_method` and has the following parameters
* `loss::Function` - The loss function. No default.
* `d_in` - The starting sampling distribution. No default.
* `max_iter` - Maximum number of iterations, No default.
* `N` - The popultation size. Default: `100`.
* `elite_thresh` - The threshold below which a sample will be considered elite. To have a fixed number of elite samples set this to `-Inf` and use the `min_elite_samples` parameter. Default: `-0.99`.
* `min_elite_samples` - The minimum number of elite samples. Default: `Int64(floor(0.1*N))`.
* `max_elite_samples` - The maximum number of allowed elite samples.  Default: `typemax(Int64)`.
* `weight_fn` - A function that specifies the weight of each sample. Use the likelihood ratio when trying to perform importance sampling. Default `(d,x) -> 1`.
* `rng::AbstractRNG` - The random number generator used. Default: `Random.GLOBAL_RNG`.
* `verbose` - Wether or not to print progress. Default: `false`
* `add_entropy` - A function that transforms the sampling distribution after fitting. Use it to enforce a maximum level of entropy if converging too quickly. Default: `(x)->x`.



