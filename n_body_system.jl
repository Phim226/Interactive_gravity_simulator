using DifferentialEquations
using Distributions
using GLMakie
using CommonSolve: init, step!

const fps = 45
#const G = 4*(pi^2) # gravitational constant in the units [au^3][solar mass^-1][sidereal year^-2]
G = 4*(pi^2)
steps_per_frame = 250
colours = [:black, :red, :blue, :green]

num_particles = rand(2:4) # generate random number of particles between 2 and 4
num_particles = 3
particles = Vector{Point2f}([])
init_pos = Vector{Float64}([])
init_vel = Vector{Float64}([])
params = Vector{Float64}([num_particles, G])
for i in 1:num_particles
    append!(init_pos, (rand((Uniform(1.0, 3.0))), rand(Uniform(1.0, 3.0))))
    #append!(init_vel, rand(Uniform(-7.5, 7.5), 2))
    append!(init_vel, (0.0,0.0))
    push!(particles, Point2f(init_pos[1], init_pos[2]))
    #push!(params, rand(Uniform(1.0, 5.0))) # add random particle weights to parameters,
    push!(params, 1.0)
    @show params[i+2]
end

num_particles = 3
params = [num_particles, G, 1, 1, 1]
init_pos = [-0.97000436, 0.24308753, 0.0, 0.0, 0.97000436, -0.24308753]
init_vel = 6.3*[0.4662036850, 0.4323657300, -0.93240737, -0.86473146, 0.4662036850, 0.4323657300]
particles = [Point2f(init_pos[2*i-1], init_pos[2*i]) for i in 1:num_particles]

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
            d = ((u[2*i-1]-u[2*j-1])^2+(u[2*i]-u[2*j])^2)^(3/2)
            r[i,j]= d
            r[j,i]= d
        end
    end

    #acceleration equations 
    for i in 1:n
        indices = [x for x in 1:n]
        deleteat!(indices, i)
        dv[2*i-1] = G*sum(p[j+2]*((u[2*j-1]-u[2*i-1])/r[i,j]) for j in indices) # dvxi/dt = G Σmi(xj-xi)/rij
        dv[2*i] = G*sum(p[j+2]*((u[2*j]-u[2*i])/r[i,j]) for j in indices) # dvyi/dt = G Σmi(yj-yi)/rij
    end
end

function fv(du, v, u, p, t)
    # v = [vx1, vy1, vx2, vy2,..., vxn, vyn]
    # p = [num_particles, G, m1, m2,..., mn]
    n = Int.(p[1])

    # velocity equations
    for i in 1:n
        du[2*i-1] = v[2*i-1] # dxi/dt = vxi
        du[2*i] = v[2*i] # dyi/dt = vyi
    end
end

tspan = 0.00005
prob = DynamicalODEProblem(fa, fv, init_vel, init_pos, (0.0, tspan), params)
integ = init(prob, KahanLi6(), dt = tspan)

markersize = [params[i+2]*18 for i in 1:num_particles]
colour = colours[1:num_particles]

fig = Figure(); display(fig)
ax = Axis(fig[1,1], limits = (-2.5,2.5,-2.5,2.5))
scatter!(ax, particles; marker = :circle,
    color = colour, markersize = markersize
)

for i in 1:200
    for j in 1:steps_per_frame
        step!(integ)
    end
    particles[] = [Point2f(integ.u[2*i+2*num_particles-1], integ.u[2*i+2*num_particles]) for i in 1:num_particles]
    #autolimits!(ax)
    sleep(1/fps)
end