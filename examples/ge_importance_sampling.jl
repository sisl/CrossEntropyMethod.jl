using CrossEntropyMethod
using Distributions, Random

base_dist = Categorical([0.74899, 0.2, 0.05, 0.001, 0.00001])
is_dist_0 = Dict{Symbol, Vector{Sampleable}}(:x => [Categorical(5)])

function l(d, s)
    v = s[:x][1]
    -(v in [3,4,5])
end

function w(d, s)
    v = s[:x][1]
    exp(logpdf(base_dist, v) - logpdf(d, s))
end

# Example.
# s = Dict(:x => [5])
# w(is_dist_0, s)

h = add_categorical_entropy([0, 0, 0, 0, 1e-10])
is_dist_opt = cross_entropy_method(l, is_dist_0; max_iter=10, N=10000, #=weight_fn=w,=# verbose=true, add_entropy=h)

display(is_dist_opt[:x][1].p)



# GE.
mu = [180, 50, -88.5, 150] # wpt-dir, wpt-dist, wind-dir, wind-mag
Sigma = diagm([45, 30, 39.5, 80])
distr = MvNormal(mu, Sigma)
is_dist_ge_0 = Dict{Symbol, Vector{Sampleable}}(:wpt => [distr])

function cem_loss(d, s; sim)
	v = s[:wpt][1]
	sample_waypoint_direction = v[1]
	sample_waypoint_distance = v[2]
	sample_wind_direction = v[3]
	sample_wind_magnitude = v[4]

	next_waypoint, next_windpoint = dist2waypoint(sim, sample_waypoint_direction, sample_waypoint_distance, sample_wind_direction, sample_wind_magnitude)
	push!(sim.waypoint, next_waypoint)
	push!(sim.windpoint, next_windpoint)

	BlackBox.evaluate(sim)

	return BlackBox.miss_distance(sim)
end