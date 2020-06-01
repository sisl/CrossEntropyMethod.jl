module CrossEntropyMethod
    export cross_entropy_method


    using Distributions
    using Random

    function Base.rand(rng::AbstractRNG, d::Dict{Symbol, Vector{Sampleable}})
        Dict(k => rand.(rng, d[k]) for k in keys(d))
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

    function Distributions.fit(d::Dict{Symbol, Vector{Sampleable}}, samples)
        N = length(samples)
        new_d = Dict{Symbol, Vector{Sampleable}}()
        for s in keys(d)
            dtype = typeof(d[s][1])
            m = length(d[s])
            new_d[s] = [fit(dtype, [samples[j][s][i] for j=1:N]) for i=1:m]
        end
        d = new_d
    end

    function Distributions.fit(d::Dict{Symbol, Tuple{Sampleable, Int64}}, samples)
        N = length(samples)
        new_d = Dict{Symbol, Tuple{Sampleable, Int64}}()
        for s in keys(d)
            dtype = typeof(d[s][1])
            m = d[s][2]
            new_d[s] = (fit(dtype, vcat([samples[j][s][:] for j=1:N]...)), m)
        end
        d = new_d
    end

    # This version uses a vector of distributions for sampling
    # N is the number of samples taken
    # m is the length of the vector
    function cross_entropy_method(loss::Function,
                                  d;
                                  max_iter,
                                  N=100,
                                  N_elite = 10,
                                  rng::AbstractRNG = Random.GLOBAL_RNG,
                                  verbose = false
                                 )
        for iteration=1:max_iter
            verbose && println("iteration ", iteration, " of ", max_iter)

            # Get samples -> Nxm
            samples = rand(rng, d, N)

            # sort the samples by elite
            order = sortperm(loss.(samples))

            #update based on elite samples
            d = fit(d, samples[order[1:N_elite]])
        end
        d
    end


end

