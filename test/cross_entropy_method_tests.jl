using CrossEntropyMethod
using Test
using Distributions
using Random
Random.seed!(0)

## Start by testing the algorithms with vectors of distributions
# Construct a distribution
d = Dict{Symbol, Vector{Sampleable}}(
    :x => fill(Categorical(5), 3),
    :y => fill(Normal(0,1), 3)
)

# Construct a loss function
function l(d, s)
    x, y = s[:x], s[:y]
    (x[1] >= 3) + (x[2] != 3) + (x[3] <= 3) + (y[1] >= 2) + (y[2] < -0.2) + (y[2] > 0.2) + (y[3] <= 2)
end

N = 100
N_elite = 10
s = rand(Random.GLOBAL_RNG, d)
logpdf(d, s)
@test logpdf(d, s) < 0
@test length(s[:x]) == 3
@test length(s[:y]) == 3
@test s[:x][1] isa Int
@test s[:y][1] isa Float64

samples = rand(Random.GLOBAL_RNG, d, N)
@test length(samples) == N
@test samples[1] isa Dict{Symbol, Vector}

avg_loss = sum([l(d, s) for s in samples])/N


order = sortperm([l(d, s) for s in samples])
elite_samples = samples[order[1:N_elite]]
@test length(elite_samples) == N_elite

new_d = fit(d, elite_samples, ones(length(elite_samples)))


new_loss = sum([l(d, s) for s in rand(Random.GLOBAL_RNG, new_d, N)])/N
@test new_loss < avg_loss

d = cross_entropy_method(l, d, max_iter=10, N=1000, min_elite_samples=100, max_elite_samples=100, verbose=false)

new_loss = sum([l(d, s) for s in rand(Random.GLOBAL_RNG, d, N)])/N
@test new_loss <= 0.03

# Now test singular distribution

d = Dict{Symbol, Tuple{Sampleable, Int64}}(
    :x => (Categorical(5), 3),
    :y => (Normal(0,1), 3)
)

# Construct a loss function
function l(d, s)
    x, y = s[:x], s[:y]
    (x[1] >= 3) + (x[2] >= 3) + (x[3] >= 3) + (y[1] >= 0)
end

N = 100
N_elite = 10
s = rand(Random.GLOBAL_RNG, d)
@test logpdf(d, s) < 0
@test length(s[:x]) == 3
@test length(s[:y]) == 3
@test s[:x][1] isa Int
@test s[:y][1] isa Float64

samples = rand(Random.GLOBAL_RNG, d, N)
@test length(samples) == N
@test samples[1] isa Dict{Symbol, Vector}

avg_loss = sum([l(d, s) for s in samples])/N

order = sortperm([l(d, s) for s in samples])
elite_samples = samples[order[1:N_elite]]
@test length(elite_samples) == N_elite

new_d = fit(d, elite_samples, ones(length(elite_samples)))
new_loss = sum([l(d,s) for s in rand(Random.GLOBAL_RNG, new_d, N)])/N
@test new_loss < avg_loss

d = cross_entropy_method(l, d, max_iter=100, N=1000, min_elite_samples=100, max_elite_samples=100, verbose=false)

new_loss = sum([l(d, s) for s in rand(Random.GLOBAL_RNG, d, N)])/N
@test new_loss <= 0.01

d = Dict{Symbol, Tuple{Sampleable, Int64}}(
    :x => (Categorical(1), 3),
)
s = rand(Random.GLOBAL_RNG, d)
@test logpdf(d, s) == 0


# Test all zero weights
d = Dict{Symbol, Tuple{Sampleable, Int64}}(
    :x => (Categorical(5), 3),
    :y => (Categorical(5), 3)
)
cross_entropy_method(l, d, max_iter=1, weight_fn=(d,x)->0.0)


# Batch testing
batch_loss = (d, X) -> begin
    # Collect each :x sample from the time-series vector X
    𝐗 = map(Xᵢ->Xᵢ[:x][1], X)

    # Squared loss (minimum at 5)
    return (𝐗 .- 5) .^ 2
end

# Initial proposal distribution
Pₒ = Dict{Symbol, Vector{Sampleable}}(:x => [Normal(0, 10)])

P′ = cross_entropy_method(batch_loss, Pₒ, max_iter=10, batched=true)

@test P′[:x][1].μ ≈ 5.0
