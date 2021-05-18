using LinearAlgebra

struct Robot
    tₛ::Float64
    A::Matrix{Float64}
    B::Matrix{Float64}
end

function parts_inds(h, nᵣ, mᵣ, nₕ, mₕ)
    n = nᵣ + nₕ + mᵣ # number of elements for each timestep
    x_inds = [(k-1)*n .+ (1:nᵣ) for k=1:h]
    x̂_inds = [(k-1)*n .+ (nᵣ+1:nᵣ+nₕ) for k=1:h] 
    u_inds = [(k-1)*n .+ (nᵣ+nₕ+1:n) for k=1:h-1]
    return [x_inds, x̂_inds, u_inds]
end

function Q_inds(h, nᵣ, mᵣ, nₕ, mₕ)
    n = nᵣ + nₕ + mᵣ
    inds = [(k-1)*n .+ (1:n) for k=1:h-1]
    end_inds = (h-1)*(nᵣ+nₕ+mᵣ) .+ (1:(nᵣ+nₕ))
    push!(inds, end_inds)
    return inds
end

function Σ_inds(h, nᵣ, mᵣ, nₕ, mₕ)
    # splitting msee into blocks
    φ_size = nₕ + mₕ
    return [(k-1)*φ_size .+ (1:φ_size) for k=1:nₕ]
end

function game_logic(prob, points)
    if norm(prob.x₀[1:2] - prob.robot_goal[1:2], 2) < 0.5
        points[1] += 1
        prob.robot_goal .= [(20 .*rand(2)) .- 10; 0; 0]
    end
    if norm(prob.x̂₀[1:2] - prob.goal[1:2], 2) < 0.5
        points[2] += 1
        prob.goal .= [(20 .*rand(2)) .- 10; 0; 0]
    end
end

function nonlinear_human(prob, xₕ, xᵣ)
    tₛ = prob.tₛ 
    Aₕ = [1 0 tₛ 0;
          0 1 0 tₛ;
          0 0 1 0;
          0 0 0 1]

    B1 = 1.0 .*[1 0 0 0;
                 0 1 0 0; 
                 0 0 0 0;
                 0 0 0 0]
    B2 = (-0.1) .*[1 0 0 0;  
                   0 1 0 0; 
                   0.5 0 0 0;
                   0 0.5 0 0]
    Bₕ = [B1 B2]

    inv_dist = 1 / norm(xₕ[1:2] - xᵣ[1:2])   
    uₕ = [inv_dist^2 .* (xₕ - xᵣ); xₕ - prob.goal]
    
    return Aₕ*xₕ + Bₕ*uₕ
end

struct BeliefCVX
    robot
    sizes
    nᵣ
    mᵣ
    nₕ
    mₕ
    h
    umax   
    umin   
    dmin
    dmax
    x₀
    xf
    uref         
    x̂₀    
    goal
    robot_goal   
    θ₀     
    θ_err₀ 
    msee_θ₀
    msee_x₀
    F₀     
    λ      
    λ₀     
    kϕ     
    dθ     
    W      
    η      
    tₛ
    Z_inds
    Q_inds
    Σ_inds
    function BeliefCVX(robot, h; nᵣ, mᵣ, nₕ, mₕ, umax, dmin, dmax, x₀, xf, uref, x̂₀, goal, robot_goal, θ₀, θ_err₀, msee_θ₀, msee_x₀, F₀, 
            λ=0.997, tₛ=0.05, λ₀=1.0, kϕ=2, dθ, W, η=10)
        sizes = (nᵣ, mᵣ, nₕ, mₕ)
        Zinds = parts_inds(h, nᵣ, mᵣ, nₕ, mₕ)
        Qinds = Q_inds(h, nᵣ, mᵣ, nₕ, mₕ)
        Σinds = Σ_inds(h, nᵣ, mᵣ, nₕ, mₕ)
        
        return new(robot, sizes, nᵣ, mᵣ, nₕ, mₕ, h, umax, -umax, dmin, dmax, x₀, xf, uref, x̂₀, goal, robot_goal, θ₀, θ_err₀, msee_θ₀, msee_x₀,
                    F₀, λ, λ₀, kϕ, dθ, W, η, tₛ, Zinds, Qinds, Σinds)
    end
end

function initialize_prob(;h=10)
    h = h
    nᵣ = 4
    nₕ = 4
    mᵣ = 2
    mₕ = 8
    tₛ = 0.05
    θ_dim = nₕ*(nₕ + mₕ)
    umax = 10
    dmin = 2
    dmax = 50
    x₀ = Array{Float64}([-5; 0; 0; 0])
    x̂₀ = Array{Float64}([5; 0; 0; 0])
    goal = Array{Float64}([0; 0; 0; 0])
    robot_goal = Array{Float64}([0; 0; 0; 0])
    xf = copy(robot_goal)
    uref = zeros(mᵣ, h-1)
    θ₀ = Array{Float64}(vec([1.0*I zeros(nₕ, mₕ)]'))
    dθ = ones(1, θ_dim) ./ 1000
    θ_err₀ = zeros(size(dθ)) #copy(dθ)
    msee_θ₀ = Array{Float64}(0.01 .*I(θ_dim))
    msee_x₀ = Array{Float64}(0.0*I(nₕ))
    F₀ = Array{Float64}(1.0*I(θ_dim))
    # W = diagm([0.1, 0.1, 0.5, 0.5])
    W = [0.00989175  0.00120164  0.0104287   0.00471507;
         0.00120164  0.00540861  0.00260211  0.00101751;
         0.0104287   0.00260211  0.0171352   0.00821844;
         0.00471507  0.00101751  0.00821844  0.00437818]
    A = [1 0 tₛ 0;
         0 1 0 tₛ;
         0 0 1 0;
         0 0 0 1]
    B = [0.5*(tₛ^2) 0; 
         0 0.5*(tₛ^2);
         tₛ 0;
         0 tₛ]
    robot = Robot(tₛ, A, B)
    prob = BeliefCVX(robot, h, nᵣ=nᵣ, mᵣ=mᵣ, nₕ=nₕ, mₕ=mₕ, umax=umax, dmin=dmin, dmax=dmax, x₀=x₀, xf=xf, uref=uref, x̂₀=x̂₀, goal=goal,
            robot_goal=robot_goal, θ₀=θ₀, dθ=dθ, θ_err₀=θ_err₀, msee_θ₀=msee_θ₀, msee_x₀=msee_x₀, F₀=F₀, W=W, η=1, tₛ=tₛ)
    return prob
end