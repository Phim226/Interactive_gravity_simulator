using DifferentialEquations: DynamicalODEProblem, DiscreteCallback, ContinuousCallback, VectorContinuousCallback, KahanLi8, get_du
using Distributions: Uniform
using GLMakie
using LinearAlgebra: norm
using CommonSolve: init, step!
using DataStructures: CircularBuffer

const fps = 30
G = 1 # gravitational constant with the units [au^3][solar mass^-1][sidereal year^-2] is 4π^2
steps_per_frame = 250
colours = [:black, :red, :blue, :green]

num_particles = rand(2:4) # generate random number of particles between 2 and 4
particles = Vector{Point2f}([])
init_pos = Vector{Float64}([])
init_vel = Vector{Float64}([])
params = Vector{Float64}([num_particles, G])
for i in 1:num_particles
    append!(init_pos, (rand((Uniform(-1.0, 1.0))), rand(Uniform(-1.0, 1.0))))
    append!(init_vel, rand(Uniform(-1.0, 1.0), 2))
    #append!(init_vel, (0.0,0.0))
    push!(particles, Point2f(init_pos[1], init_pos[2]))
    #push!(params, rand(Uniform(1.0, 5.0))) # add random particle weights to parameters,
    push!(params, 1.0) # weight of particle
end

# Manually input initial conditions here 
#= num_particles = 3
params = [3, G, 1, 1, 1] 
init_pos = [-0.15934774719487366, 0.7791598189274951, 0.37322805777650814, 0.7311623519644381, -0.6840866845336071, -0.9943633409754717]
init_vel = [-0.7880389812714281, 0.27170588390371564, -0.48427701549690827, 0.8305093388782598, 0.8440809991793201, 0.35358153764201217]
particles = [Point2f(init_pos[2i-1], init_pos[2i]) for i in 1:num_particles] =#

@show num_particles

particles = Observable(particles)

function fa(dv, v, u, p, t)
    # u = [x1, y1, x2, y2,..., xn, yn]
    # p = [num_particles, G, m1, m2,..., mn]
    n = Int.(p[1])
    G = p[2]

    # Distances between each of the particles. The nested loop first calculates r12, r13,..., r1n, 
    # then r23, r24,..., r2n, and so on until finally r(n-1)n where rij is the distance between
    # particles i and j 
    r = zeros(n,n)
    for i in 1:n-1
        for j in i+1:n
            d = ((u[2i-1]-u[2j-1])^2+(u[2i]-u[2j])^2)^(3/2)
            r[i,j]= d
            r[j,i]= d
        end
    end

    #acceleration equations 
    for i in 1:n
        indices = [x for x in 1:n]
        deleteat!(indices, i)
        dv[2*i-1] = G*sum(p[j+2]*((u[2j-1]-u[2i-1])/r[i,j]) for j in indices) # dvxi/dt = G Σmi(xj-xi)/rij
        dv[2*i] = G*sum(p[j+2]*((u[2j]-u[2i])/r[i,j]) for j in indices) # dvyi/dt = G Σmi(yj-yi)/rij
    end
end

function fv(du, v, u, p, t)
    # v = [vx1, vy1, vx2, vy2,..., vxn, vyn]
    # p = [num_particles, G, m1, m2,..., mn]
    n = Int.(p[1])

    # velocity equations
    for i in 1:n
        du[2i-1] = v[2i-1] # dxi/dt = vxi
        du[2i] = v[2i] # dyi/dt = vyi
    end
end

function condition(u, t, integ)
    dv = [norm([get_du(integ)[2i-1], get_du(integ)[2i]]) for i in 1:num_particles]
    accel_sum = sum(dv)
    accel_bool=((accel_sum>traj_res_treshold) && (traj_update_per_frame==low_res_traj_update)) || ((accel_sum<traj_res_treshold) && (traj_update_per_frame==high_res_traj_update))
    return accel_bool
end

function affect!(integ)
    if traj_update_per_frame==high_res_traj_update
        global traj_update_per_frame=low_res_traj_update
    elseif traj_update_per_frame==low_res_traj_update
        global traj_update_per_frame=high_res_traj_update
    end
    global traj_update_partition = [i*(trunc(Int, steps_per_frame/traj_update_per_frame)) for i in 1:traj_update_per_frame]
end

cb = DiscreteCallback(condition, affect!)

tspan = 0.0001
prob = DynamicalODEProblem(fa, fv, init_vel, init_pos, (0.0, tspan), params)
integ = init(prob, KahanLi8(), callback = cb, dt = tspan)

traj_length = 2000
trajectories = []
for i in 1:num_particles
    traj = CircularBuffer{Point2f}(traj_length)
    fill!(traj, Point2f(init_pos[2i-1], init_pos[2i]))
    #traj = Vector{Point2f}([Point2f(init_pos[2i-1], init_pos[2i])])
    traj = Observable(traj)
    push!(trajectories, traj)
end

markersize = [params[i+2]*18 for i in 1:num_particles]
colour = colours[1:num_particles]

fig = Figure(); display(fig)
ax = Axis(fig[1,1], limits = (-1.5,1.5,-1.5,1.5))
scatter!(ax, particles; marker = :circle,
    color = colour, markersize = markersize
)

for i in 1:num_particles
    col = to_color(colour[i])
    traj_colour = [RGBAf(col.r, col.g, col.b, (i/traj_length)^2) for i in 1:traj_length]
    lines!(ax, trajectories[i]; linewidth = 2, color = traj_colour)
end

low_res_traj_update = 1
high_res_traj_update = 15
traj_res_treshold = 50
traj_update_per_frame = low_res_traj_update
traj_update_partition = [i*(trunc(Int, steps_per_frame/traj_update_per_frame)) for i in 1:traj_update_per_frame]

function run_sim(frames)
    for i in 1:frames
        p_index = 1
        for j in 1:steps_per_frame
            step!(integ)
            if length(traj_update_partition)==1
                p_index=1
            end
            if j==traj_update_partition[p_index]
                for n in 1:num_particles
                    push!(trajectories[n][], Point2f(integ.u[2n+2num_particles-1], integ.u[2n+2num_particles]))
                    trajectories[n][] = trajectories[n][]
                end
                if p_index<traj_update_per_frame
                    p_index +=1
                end
            end
        end
        particles[] = [Point2f(integ.u[2i+2num_particles-1], integ.u[2i+2num_particles]) for i in 1:num_particles]
        #= println("--------------")
        @show integ.t
        println("")
        dv1 = [get_du(integ)[1], get_du(integ)[2]]
        dv2 = [get_du(integ)[3], get_du(integ)[4]]
        dv3 = [get_du(integ)[5], get_du(integ)[6]]
        du1 = [get_du(integ)[7], get_du(integ)[8]]
        du2 = [get_du(integ)[9], get_du(integ)[10]]
        du3 = [get_du(integ)[11], get_du(integ)[12]]
        @show norm(dv1)
        @show norm(du1)
        println("")
        @show norm(dv2)
        @show norm(du2)
        println("")
        @show norm(dv3)
        @show norm(du3)
        println("")
        @show traj_update_per_frame
        println("--------------") =#
        sleep(1/fps)
    end
end

frames = 500 # run the simulation for this number of frames
run_sim(frames)