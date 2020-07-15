module CrossEntropyMethod
    using Distributions
    using Random
    using POMDPs
    using POMDPSimulators
    using POMDPPlayback

    using POMDPs, POMDPPlayback, POMDPModels, POMDPPolicies, Distributions, Random
    using Test

    export cross_entropy_method, add_categorical_entropy
    include("cross_entropy_method.jl")

    export mdp_loss, mdp_weight, continous_actions_weight, sample_playback_policy_fn
    include("mdp_helpers.jl")
end

