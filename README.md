# CrossEntropyMethod.jl
[![Build Status](https://travis-ci.org/sisl/CrossEntropyMethod.jl.svg?branch=master)](https://travis-ci.org/sisl/CrossEntropyMethod.jl) [![Coverage Status](https://coveralls.io/repos/github/sisl/CrossEntropyMethod.jl/badge.svg?branch=master)](https://coveralls.io/github/sisl/CrossEntropyMethod.jl?branch=master) [![codecov](https://codecov.io/gh/sisl/CrossEntropyMethod.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/sisl/CrossEntropyMethod.jl)

This package provides an implementation of the cross entropy method for optimizing multivariate time series distributions.
Suppose we have a timeseries `X = {x₁, ..., xₙ}` where each `xᵢ` is a vector of dimension `m`. This package provides optimization for two different scenarios:
1. The time series is sampled IID from a single distribution `p`: `xᵢ ~ p(x)`. In this case, the distribution is represented as a `Dict{Symbol, Tuple{Sampleable, Int64}}`. The dictionary will contain `m` symbols, one for each variable in the series. The `Sampleable` object represents `p` and the integer is the length of the timeseries (`N`)
2. The time series is sampled from a different distribution at each timestep `pᵢ`: `xᵢ ~ pᵢ(x)`. In this case, the distribution is also represented as a `Dict{Symbol, Tuple{Sampleable, Int64}}`.

Note: The `Sampleable` objects must support the `Distributions.jl` function `logpdf` and `fit`.

## Usage
See the `examples/` folder for an example use case.
The main function is `cross_entropy_method` and has the following parameters:
* `loss::Function` - The loss function. No default.
* `d_in` - The starting sampling distribution. No default.
* `max_iter` - Maximum number of iterations, No default.
* `N` - The population size. Default: `100`
* `elite_thresh` - The threshold below which a sample will be considered elite. To have a fixed number of elite samples set this to `-Inf` and use the `min_elite_samples` parameter. Default: `-0.99`
* `min_elite_samples` - The minimum number of elite samples. Default: `Int64(floor(0.1*N))`
* `max_elite_samples` - The maximum number of allowed elite samples.  Default: `typemax(Int64)`
* `weight_fn` - A function that specifies the weight of each sample. Use the likelihood ratio when trying to perform importance sampling. Default `(d,x) -> 1`
* `rng::AbstractRNG` - The random number generator used. Default: `Random.GLOBAL_RNG`
* `verbose` - Whether or not to print progress. Default: `false`
* `show_progress` - Whether or not to show the progress meter. Default: `false`
* `batched` - Indicates batched loss evaluation (loss function must return an array containing loss values for each sample). Default: `false`
* `add_entropy` - A function that transforms the sampling distribution after fitting. Use it to enforce a maximum level of entropy if converging too quickly. Default: `(x)->x`



Maintained by Anthony Corso (acorso@stanford.edu)
