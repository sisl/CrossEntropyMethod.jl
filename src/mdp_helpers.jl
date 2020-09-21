function get_playback_policy(d, s, mdp, backup_policy)
    logpdfs = [logpdf(d, s, i) for i=1:length(first(s)[2])]
    PlaybackPolicy([a for a in actions(mdp)[s[:a]]], backup_policy, logpdfs, 1)
end

function sample_playback_policy_fn(d, mdp, backup_policy, rng=Random.GLOBAL_RNG)
    () -> get_playback_policy(d, rand(rng, d), mdp, backup_policy)
end

function mdp_loss(mdp, backup_policy)
    function loss(d, s)
        p = get_playback_policy(d, s, mdp, backup_policy)
        -simulate(RolloutSimulator(), mdp, p)
    end
end

function mdp_weight(mdp)
    function weight(d, s)
        exp(logpdf(mdp, actions(mdp)[s[:a]]) - logpdf(d, s))
    end
end

function continous_actions_weight(d_true)
    function weight(d, s)
        exp(logpdf(d_true, s) - logpdf(d, s))
    end
end


