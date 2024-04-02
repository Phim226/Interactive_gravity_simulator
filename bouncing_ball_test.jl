using DifferentialEquations
using GLMakie
using CommonSolve: init, step!

const fps = 30

function f(du, u, p, t)
    du[1] = u[3] #dx/dt = vx
    du[2] = u[4] #dy/dt = vy
    du[3] = 0 #dvx/dt = 0 
    du[4] = -p # dvy/dt = -g 
end

function condition(u, t, integrator) # Event when condition(u,t,integrator) == 0
    u[2]
end

function affect!(integrator)
    integrator.u[4] = -0.9*integrator.u[4]
end

cb = ContinuousCallback(condition, affect!)

u0 = [0.1, 0.5, 0.1, 0.0]
tspan = 0.05
p = 9.81
prob = ODEProblem(f, u0, tspan, p)
integ = init(prob, Tsit5(), callback = cb)

ball = Observable(Point2f(0.1,0.5))
floor = [Point2f(0,0), Point2f(100,0)]

fig = Figure(); display(fig)
ax = Axis(fig[1,1], limits = (0,1,0,1))
lines!(ax, floor; linewidth = 4, color = :purple)

# markersize is the radius pixels of the marker, therefore the boundary of the disc is 12 pixels
# from the position of the disc if markersize = 12
scatter!(ax, ball; marker = :circle,
    color = :black, markersize = 12
)

ball[] = Point2f(0.1, 0.5)
for i in 1:200
    @show integ.u, integ.t
    step!(integ)
    ball[] = Point2f(integ.u[1], integ.u[2])
    #autolimits!(ax)
    sleep(1/fps)
end