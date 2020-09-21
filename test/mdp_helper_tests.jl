using CrossEntropyMethod
using POMDPs
using POMDPModels
using POMDPPolicies
using Distributions
using Random
using Test

mdp = SimpleGridWorld(tprob = 1)

backup_policy = RandomPolicy(mdp)
Distributions.logpdf(mdp::SimpleGridWorld, h) = length(h)*log(0.25)
Distributions.logpdf(p::RandomPolicy, h) = length(h)*log(0.25)

d = Dict{Symbol, Tuple{Sampleable, Int64}}(:a => (Categorical(4), 30))
loss = mdp_loss(mdp, backup_policy)
d_opt = cross_entropy_method(loss, d, max_iter = 10, N=1000)
@test argmax(d_opt[:a][1].p) == 4

mdp = SimpleGridWorld(tprob = 1, rewards = Dict(GWPos(1,1)=>0., GWPos(10,10)=>1.))
loss = mdp_loss(mdp, backup_policy)
weight = mdp_weight(mdp)
h = add_categorical_entropy([1., 1., 1., 1.])
d_opt = cross_entropy_method(loss, d, max_iter = 10, N=100, weight_fn = weight, add_entropy = h)

@test argmax(d_opt[:a][1].p) == 4 || argmax(d_opt[:a][1].p) == 1
d_opt[:a][1].p


