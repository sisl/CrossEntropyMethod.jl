using CrossEntropyMethod
using Test
using Distributions
using Random

## Start by testing the algorithms with vectors of distributions
# construct a distribution
d = Dict{Symbol, Vector{Sampleable}}(
    :x => fill(Categorical(5), 3),
    :y => fill(Normal(0,1), 3)
)

#construct a loss function
function loss(d)
    x, y = d[:x], d[:y]
    (x[1] >= 3) + (x[2] != 3) + (x[3] <= 3) + (y[1] >= 2) + (y[2] < -0.2) + (y[2] > 0.2) + (y[3] <= 2)
end

N = 100
N_elite = 10
s = rand(Random.GLOBAL_RNG, d)
@test length(s[:x]) == 3
@test length(s[:y]) == 3
@test s[:x][1] isa Int
@test s[:y][1] isa Float64

samples = rand(Random.GLOBAL_RNG, d, N)
@test length(samples) == N
@test samples[1] isa Dict{Symbol, Vector}

avg_loss = sum(loss.(samples))/N


order = sortperm(loss.(samples))
elite_samples = samples[order[1:N_elite]]
@test length(elite_samples) == N_elite

new_d = fit(d, elite_samples)


new_loss = sum(loss.(rand(Random.GLOBAL_RNG, new_d, N)))/N
@test new_loss < avg_loss

d = cross_entropy_method(loss, d, max_iter =10, N=1000, N_elite = 100, verbose = true)

new_loss = sum(loss.(rand(Random.GLOBAL_RNG, d, N)))/N
@test new_loss == 0

## Now test singular distribution

d = Dict{Symbol, Tuple{Sampleable, Int64}}(
    :x => (Categorical(5), 3),
    :y => (Normal(0,1), 3)
)

#construct a loss function
function loss(d)
    x, y = d[:x], d[:y]
    (x[1] >= 3) + (x[2] >= 3) + (x[3] >= 3) + (y[1] >= 0)
end

N = 100
N_elite = 10
s = rand(Random.GLOBAL_RNG, d)
@test length(s[:x]) == 3
@test length(s[:y]) == 3
@test s[:x][1] isa Int
@test s[:y][1] isa Float64

samples = rand(Random.GLOBAL_RNG, d, N)
@test length(samples) == N
@test samples[1] isa Dict{Symbol, Vector}

avg_loss = sum(loss.(samples))/N

order = sortperm(loss.(samples))
elite_samples = samples[order[1:N_elite]]
@test length(elite_samples) == N_elite

new_d = fit(d, elite_samples)
new_loss = sum(loss.(rand(Random.GLOBAL_RNG, new_d, N)))/N
@test new_loss < avg_loss

d = cross_entropy_method(loss, d, max_iter =100, N=1000, N_elite = 100, verbose = true)

new_loss = sum(loss.(rand(Random.GLOBAL_RNG, d, N)))/N
@test new_loss <= 0.01

