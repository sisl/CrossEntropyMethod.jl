module CrossEntropyMethod
    using Distributions
    using Random
    using ProgressMeter
    using Requires

    export cross_entropy_method, add_categorical_entropy
    include("cross_entropy_method.jl")

    # Use MDP helpers if POMDPPlayback is loaded
    function __init__()
        @require POMDPPlayback="9532798a-b2b7-41e0-9ed3-6c6560f8cd4c" begin
            using POMDPs
            using POMDPSimulators
            # using POMDPPlayback

            export mdp_loss, mdp_weight, continous_actions_weight, sample_playback_policy_fn
            include("mdp_helpers.jl")
        end
    end
end

