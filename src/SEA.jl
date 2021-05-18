using ForwardDiff
using StaticArrays
using LinearAlgebra
using Ipopt
using MathOptInterface
const MOI = MathOptInterface
using CSV, DataFrames
using PyPlot
using Distributions

function get_uᵣ_SEA(nlp, u0, ssa=false)
    # setup constants
    ∂M_∂X = SA[1 0 0 0;
               0 1 0 0;
               0 0 0 0;
               0 0 1 0;
               0 0 0 1;
               0 0 0 0]
    ∂Mᵣ_∂xᵣ = ∂M_∂X
    ∂Mₕ_∂xₕ = ∂M_∂X

    p_idx = SA[1; 2; 3]
    v_idx = SA[4; 5; 6]
    
    xᵣ = nlp.x₀
    xₕ = nlp.x̂₀
    msee_x = nlp.msee_x₀

    Mᵣ = [xᵣ[1:2]; 0; xᵣ[3:4]; 0]
    Mₕ = [xₕ[1:2]; 0; xₕ[3:4]; 0]

    ẋᵣ = zeros(nlp.nᵣ)
    ẋₕ = zeros(nlp.nₕ)
    if false
        ẋᵣ = (xᵣ - X[i-1]) ./ nlp.tₛ
        ẋₕ = (xₕ - nlp.xh[i-1]) ./ nlp.tₛ
    end
    fₓ = ((nlp.robot.A - I(4)) ./ nlp.tₛ)*xᵣ
    fᵤ = nlp.robot.B ./ nlp.tₛ

    d = norm(Mᵣ[p_idx] - Mₕ[p_idx])
    Ṁᵣ = ∂Mᵣ_∂xᵣ*ẋᵣ
    Ṁₕ = ∂Mₕ_∂xₕ*ẋₕ

    dM = Mᵣ - Mₕ
    dṀ = Ṁᵣ - Ṁₕ
    dp = dM[p_idx]
    dv = dM[v_idx]
    dṗ = dṀ[p_idx]
    dv̇ = dṀ[v_idx]

    #ḋ is the component of velocity lies in the dp direction
    ḋ = dp'dv ./ d

    # partial derivatives of ḋ wrt dM
    ∂ḋ_∂dp = (dv ./ d) - ((dp'dv).*dp ./ (d^3))
    ∂ḋ_∂dv = dp ./ d

    # partial derivatives of dp
    ∂dp_∂Mᵣ = [I(3) zeros(3, 3)]
    ∂dp_∂Mₕ = -∂dp_∂Mᵣ

    # partial derivatives of dv
    ∂dv_∂Mᵣ = [zeros(3, 3) I(3)]
    ∂dv_∂Mₕ = -∂dv_∂Mᵣ

    # partial derivatives of ḋ
    ∂ḋ_∂Mᵣ = ∂dp_∂Mᵣ'∂ḋ_∂dp + ∂dv_∂Mᵣ'∂ḋ_∂dv
    ∂ḋ_∂Mₕ = ∂dp_∂Mₕ'∂ḋ_∂dp + ∂dv_∂Mₕ'∂ḋ_∂dv
    ∂ḋ_∂xᵣ = ∂Mᵣ_∂xᵣ'∂ḋ_∂Mᵣ
    ∂ḋ_∂xₕ = ∂Mₕ_∂xₕ'∂ḋ_∂Mₕ

    # partial derivatives of d
    ∂d_∂Mᵣ = [dp ./ d; zeros(3, 1)]
    ∂d_∂Mₕ = [-dp ./ d; zeros(3, 1)]
    ∂d_∂xᵣ = reshape(∂Mᵣ_∂xᵣ'∂d_∂Mᵣ, nlp.nᵣ)
    ∂d_∂xₕ = reshape(∂Mₕ_∂xₕ'∂d_∂Mₕ, nlp.nₕ)

    # partial derivatives of ϕ
    ∂ϕ_∂xᵣ = -2*d.*∂d_∂xᵣ - nlp.kϕ.*∂ḋ_∂xᵣ
    ∂ϕ_∂xₕ = -2*d.*∂d_∂xₕ - nlp.kϕ.*∂ḋ_∂xₕ

    # calculating λ_SEA
    if ssa
        λ_SEA = nlp.λ₀
    else
        λ_SEA = 3/nlp.tₛ * sqrt(∂ϕ_∂xₕ'msee_x*∂ϕ_∂xₕ) + nlp.λ₀
    end

    # calculating ϕ
    ϕ = nlp.dmin^2 + (nlp.η*nlp.tₛ) + (λ_SEA*nlp.tₛ) - d^2 - (nlp.kϕ*ḋ)

    # calculating L and S
    L = ∂ϕ_∂xᵣ'fᵤ
    S = -nlp.η - λ_SEA - ∂ϕ_∂xₕ'ẋₕ - ∂ϕ_∂xᵣ'fₓ
    
    Lu = (L*u0)[1]
    if ϕ < 0 || Lu <= S
        return u0
    end
    Q_inv = I(nlp.mᵣ)
    lam = (Lu - S) / (L*Q_inv*L')
    c = lam*L*Q_inv*L'
    u = u0 - c*((Q_inv*L') / (L*Q_inv*L'))
    return vec(u)
end

function get_uref(nlp)
    K = [1.5 0 1.7 0;
         0 1.5 0 1.7]
    xᵣ = nlp.x₀
    state_goal = [nlp.robot_goal[1], nlp.robot_goal[2], 0, 0]
    return -K*(xᵣ - state_goal)
end

function get_uref(nlp, xᵣ)
    K = [1.5 0 1.7 0;
         0 1.5 0 1.7]
    state_goal = [nlp.robot_goal[1], nlp.robot_goal[2], 0, 0]
    return -K*(xᵣ - state_goal)
end

function get_uₕ(prob, xₕ, xᵣ)
    return [xᵣ - xₕ; xₕ - prob.goal]
end

function adapt_θ!(nlp, xₕ_prev, xₕ_obs, xᵣ, uᵣ)
    θ_dim = nlp.nₕ*(nlp.nₕ + nlp.mₕ)
    F_prev = nlp.F₀
    msee_θ_prev = nlp.msee_θ₀
    θ_err_prev = nlp.θ_err₀
    θ_prev = nlp.θ₀
    obs_dim = nlp.nₕ + nlp.mₕ
    mₕ2 = nlp.mₕ - nlp.nᵣ - nlp.mᵣ
    
    # constructing human control 
    uₕ = xₕ_prev - nlp.goal

    # constructing observation φ
    φ = zeros(obs_dim)
    φ[1:nlp.nₕ] .= xₕ_prev
    φ[nlp.nₕ+1:nlp.nₕ+nlp.nᵣ] .= xᵣ - xₕ_prev
    φ[nlp.nₕ+nlp.nᵣ+1:end] .= uₕ
    # constructing Φ matrix
    Φ = zeros(nlp.nₕ, θ_dim)
    for j=1:4
        Φ[j,(j-1)*obs_dim+1:(j-1)*obs_dim+obs_dim] .= φ;
    end
    
    # xₕ covariance update
    msee_x = (Φ*msee_θ_prev*Φ') + nlp.W
    
    # learning gain updates
    Fₜ = (F_prev-(F_prev*Φ'inv(nlp.λ.*I(nlp.nₕ) + Φ*F_prev*Φ')*Φ*F_prev))./nlp.λ
    
    # θ_err update
    θ_err = ((I-Fₜ*Φ'*Φ)*θ_err_prev')' + nlp.dθ
    
    # belief dynamics update
    msee_θₜ = msee_θ_prev + (Fₜ*Φ'*msee_x*Φ*Fₜ) - (msee_θ_prev*Φ'*Φ*Fₜ) - (
                Fₜ*Φ'*Φ*msee_θ_prev) + (θ_err'nlp.dθ) + (nlp.dθ'θ_err) - (nlp.dθ'nlp.dθ)
    
    ############
    α = 0.5
    x_post = (1-α).*Φ*θ_prev + α.*xₕ_obs
    x_err = x_post - Φ*θ_prev
    ############

    # x_err = xₕ_obs - Φ*θ_prev # prediction error
    θₜ = θ_prev + (Fₜ*Φ'x_err)
    
    # update nlp
    nlp.x₀ .= xᵣ
    # nlp.x̂₀ .= xₕ_obs
    nlp.x̂₀ .= x_post
    nlp.msee_x₀ .= msee_x
    nlp.F₀ .= Fₜ
    nlp.θ_err₀ .= θ_err
    nlp.msee_θ₀ .= msee_θₜ
    nlp.θ₀ .= θₜ
    
    return norm(x_err)
end

function filter_u(u)
    if norm(u) > 10
        return u .* (10 / norm(u))
    end
    return u
end

function human_dyn(nlp, human, k, xₕ, uₕ)
    goal = [5.0, 5.0]
    center = 0.5.*(goal + human.x₀[[1,2]])
    vec = xₕ[[1,2]] - center
    tan_vec = [-vec[2], vec[1]]

    dx = human.step*tan_vec[1]
    dy = human.step*tan_vec[2]
    
    x_new = [xₕ[1] + dx, xₕ[2] + dy]
    ẋ_new = (x_new - xₕ[[1,2]])./nlp.tₛ
    x = [x_new[1], x_new[2], ẋ_new[1], ẋ_new[2]]
    
    d = norm(xₕ - nlp.x₀) # distance between human and robot
    B1 = 0.5 .*[0 0 0 0;
             0 0 0 0; 
             0 0 tₛ 0;
             0 0 0 tₛ]
    x = x + (1 / d).*B1*uₕ[1:4]
    
    return x
end

struct Human2
    x₀
    dθ
    step
end

struct Human
    A
    B
end