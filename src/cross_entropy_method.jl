function Distributions.logpdf(d::Dict{Symbol, Vector{Sampleable}}, x, i)
    sum([logpdf(d[k][i], x[k][i]) for k in keys(d)])
end

function Distributions.logpdf(d::Dict{Symbol, Vector{Sampleable}}, x)
    sum([logpdf(d, x, i) for i=1:length(first(x)[2])])
end

function Distributions.logpdf(d::Dict{Symbol, Tuple{Sampleable, Int64}}, x, i)
    sum([logpdf(d[k][1], x[k][i]) for k in keys(d)])
end

function Distributions.logpdf(d::Dict{Symbol, Tuple{Sampleable, Int64}}, x)
    sum([logpdf(d, x, i) for i=1:length(first(x)[2])])
end

function Base.rand(rng::AbstractRNG, d::Dict{Symbol, Vector{Sampleable}})
    Dict(k => rand.(Ref(rng), d[k]) for k in keys(d))
end

function Base.rand(rng::AbstractRNG, d::Dict{Symbol, Tuple{Sampleable, Int64}})
    Dict(k => rand(rng, d[k][1], d[k][2]) for k in keys(d))
end

function Base.rand(rng::AbstractRNG, d::Dict{Symbol, Vector{Sampleable}}, N::Int)
    [rand(rng, d) for i=1:N]
end

function Base.rand(rng::AbstractRNG, d::Dict{Symbol, Tuple{Sampleable, Int64}}, N::Int)
    [rand(rng, d) for i=1:N]
end

function Distributions.fit(d::Dict{Symbol, Vector{Sampleable}}, samples, weights; add_entropy = (x) -> x)
    N = length(samples)
    new_d = Dict{Symbol, Vector{Sampleable}}()
    for s in keys(d)
        dtype = typeof(d[s][1])
        m = length(d[s])
        new_d[s] = [add_entropy(fit(dtype, [samples[j][s][i] for j=1:N], weights)) for i=1:m]
    end
    d = new_d
end

function Distributions.fit(d::Dict{Symbol, Tuple{Sampleable, Int64}}, samples, weights; add_entropy = (x)->x)
    N = length(samples)
    new_d = Dict{Symbol, Tuple{Sampleable, Int64}}()
    for s in keys(d)
        dtype = typeof(d[s][1])
        m = d[s][2]
        all_samples = vcat([samples[j][s][:] for j=1:N]...)
        all_weights = vcat([fill(weights[j], length(samples[j][s][:])) for j=1:N]...)
        new_d[s] = (add_entropy(fit(dtype, all_samples, all_weights)), m)
    end
    d = new_d
end

# Gets a function that adds entropy h to categorical distribution c
function add_categorical_entropy(h::Vector{Float64})
    function add_entropy(c::Categorical)
        p = h
        p[1:length(c.p)] .+= c.p
        Categorical(p ./ sum(p))
    end
end

# This version uses a vector of distributions for sampling
# N is the number of samples taken
# m is the length of the vector
function cross_entropy_method(loss::Function,
                              d_in;
                              max_iter,
                              N=100,
                              elite_thresh = -0.99,
                              min_elite_samples = Int64(floor(0.1*N)),
                              max_elite_samples = typemax(Int64),
                              weight_fn = (d,x) -> 1.,
                              rng::AbstractRNG = Random.GLOBAL_RNG,
                              verbose = false,
                              show_progress = false,
                              add_entropy = (x)->x
                             )
    d = deepcopy(d_in)
    show_progress ? progress = Progress(max_iter) : nothing

    for iteration in 1:max_iter
        verbose && @show(iteration)
        # Get samples -> Nxm
        samples = rand(rng, d, N)

        # sort the samples by loss and select elite number
        losses = [loss(d, s) for s in samples]
        order = sortperm(losses)
        losses = losses[order]
        N_elite = losses[end] < elite_thresh ? N : findfirst(losses .> elite_thresh) - 1
        N_elite = min(max(N_elite, min_elite_samples), max_elite_samples)

        verbose && println("iteration ", iteration, " of ", max_iter, " N_elite: ", N_elite)

        #update based on elite samples
        elite_samples = samples[order[1:N_elite]]
        weights = [weight_fn(d, s) for s in elite_samples]
        if all(weights .≈ 0.)
            println("Warning: all weights are zero")
        end
        d = fit(d, elite_samples, weights, add_entropy = add_entropy)
        show_progress && next!(progress)
    end
    d
end

