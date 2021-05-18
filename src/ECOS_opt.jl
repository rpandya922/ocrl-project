using Convex
using ECOS
using MathOptInterface
const MOI = MathOptInterface
include("SEA.jl")
include("BeliefCVX.jl")

function get_obj(prob)
    nᵣ, mᵣ, nₕ, mₕ = prob.sizes
    Q = zeros(nₕ+mₕ, nₕ+mₕ)
    Σ_θ = prob.msee_θ₀
    Σinds = prob.Σ_inds
    for i=1:nₕ
        for j=1:nₕ
            Σ_cij = view(Σ_θ, Σinds[i], Σinds[j])
            Q .= Q + Σ_cij 
        end
    end
    Q .= Symmetric(Q)
    return Q
end

function setup_problem(prob)
    # robot dynamics constraints
    cons = Constraint[]
    xᵣ = [Variable(prob.nᵣ) for k=1:prob.h]
    uᵣ = [Variable(prob.mᵣ) for k=1:prob.h-1]
    for k=2:prob.h
        push!(cons, xᵣ[k] == prob.robot.A*xᵣ[k-1] + prob.robot.B*uᵣ[k-1])
    end
    
    # human dynamics constraints
    C = reshape(prob.θ₀, prob.nₕ+prob.mₕ, prob.nₕ)'
    Aₕ = C[1:prob.nₕ,1:prob.nₕ]
    Bₕ = C[1:prob.nₕ,prob.nₕ+1:end]

    xₕ = [Variable(prob.nₕ) for k=1:prob.h]
    for k=1:prob.h-1
        uₕ = [xₕ[k] - xᵣ[k]; xₕ[k] - prob.goal]
        push!(cons, xₕ[k+1] == Aₕ*xₕ[k] + Bₕ*uₕ)
    end
    
    # initial state constraints
    push!(cons, xᵣ[1] == prob.x₀)
    push!(cons, xₕ[1] == prob.x̂₀)
    
    # distance constraints
    τ = [Variable(1) for k=1:prob.h-1]
    for k=1:prob.h-1
        push!(cons, norm(xᵣ[k+1] - xₕ[k+1], 2) <= τ[k])
        push!(cons, τ[k] >= prob.dmin)
    end
    
    # actuator limits
    for k=1:prob.h-1
        push!(cons, uᵣ[k] <= prob.umax*ones(prob.mᵣ))
        push!(cons, uᵣ[k] >= prob.umin*ones(prob.mᵣ))
    end
    
    # reference tracking
    uref = zeros(prob.mᵣ, prob.h-1)
    xref = zeros(prob.nᵣ, prob.h)
    xref[:,1] .= prob.x₀
    for k=1:prob.h-1
        uref[:,k] .= get_uref(prob, xref[:,k])
        xref[:,k+1] .= prob.robot.A*xref[:,k] + prob.robot.B*uref[:,k]
    end
    
    # setting active exploration objective
    Q = get_obj(prob)
    obj = [quadform([xₕ[k]; xₕ[k] - xᵣ[k]; xₕ[k] - prob.goal], Q) for k=2:prob.h]
    
    # setting reference tracking objective
    Q_ref = 10.0*I(prob.nᵣ)
    R_ref = 10.0*I(prob.mᵣ)
    Qf = 50.0*I(prob.nᵣ)
    for k=1:prob.h-1
        push!(obj, quadform(uᵣ[k] - uref[:,k], R_ref))
        push!(obj, quadform(xᵣ[k] - xref[:,k], Q_ref))
    end
    push!(obj, quadform(xᵣ[prob.h] - xref[:,prob.h], Qf))
    
    problem = minimize(sum(obj), cons)
    
    return problem, xᵣ, xₕ, uᵣ
end

function run_trial(;T=400, h=40, cont=:active, ssa=true)
    prob = initialize_prob(h=h)
    xᵣ = prob.x₀
    xₕ = prob.x̂₀
    tₛ = prob.tₛ 
    Aₕ = [1 0 tₛ 0;
          0 1 0 tₛ;
          0 0 1 0;
          0 0 0 1]

    B1 = 0.5 .*[0 0 0 0;
                 0 0 0 0; 
                 0 0 tₛ 0;
                 0 0 0 tₛ]
    B2 = (-0.1) .*[1 0 0 0;  
                   0 1 0 0; 
                   0 0 0 0;
                   0 0 0 0]
    Bₕ = [B1 B2]
    A_true = Aₕ
    B_true = Bₕ
    θ_true = vec([A_true B_true]')
    human = Human(A_true, B_true)
    points = [0, 0]
    prob.robot_goal .= [(20 .*rand(2) .- 10); 0; 0]

    # storing data
    all_xₕ = [xₕ zeros(prob.nₕ, T-1)]
    all_xᵣ = [xᵣ zeros(prob.nᵣ, T-1)]
    all_uᵣ = zeros(prob.mᵣ, T-1)
    all_msee_x = zeros(prob.nₕ, prob.nₕ, T)
    all_msee_x[:,:,1] .= prob.msee_x₀
    all_goals = [prob.goal zeros(prob.nₕ, T-1)]
    all_goals_h = [prob.goal zeros(prob.nₕ, T-1)]
    all_goals_r = [prob.robot_goal zeros(prob.nₕ, T-1)]
    all_dist = zeros(T-1)
    all_times = []

    # saving metrics
    pred_err = zeros(T-1)
    θ_est_err = zeros(T-1)
    msee_norm = zeros(T-1)
    noise = MvNormal(zeros(4), 0.1I)
    for k=2:T
        
        # extract first action for the robot
        if cont == :active
            problem, xᵣ_var, xₕ_var, uᵣ_var = setup_problem(prob)
            solve!(problem, () -> ECOS.Optimizer(verbose=false))
            if problem.status == MOI.INFEASIBLE
                uᵣ = get_uref(prob)
                r = @timed evaluate(uᵣ_var[1])
                uᵣ = r[1]
                push!(all_times, r[2])
            else
                # extract first action for the robot
                uᵣ = evaluate(uᵣ_var[1])
            end
        elseif cont == :sea
            uᵣ = get_uref(prob)
        end
        uᵣ = get_uᵣ_SEA(prob, uᵣ, ssa)

        # robot and human move
        uₕ = get_uₕ(prob, xₕ, xᵣ)
        xᵣ = prob.robot.A*xᵣ + prob.robot.B*uᵣ
        # xₕ_obs = human.A*xₕ + human.B*uₕ + rand(noise)
        # xₕ_true = human.A*xₕ + human.B*uₕ
        xₕ_true = nonlinear_human(prob, xₕ, xᵣ)
        xₕ_obs = xₕ_true + rand(noise)
        
        # do adaptation
        err = adapt_θ!(prob, prob.x̂₀, xₕ_obs, xᵣ, uᵣ)
        xₕ = xₕ_true
        
        game_logic(prob, points)
        
        # store data
        all_xₕ[:,k] .= xₕ
        all_xᵣ[:,k] .= xᵣ
        all_uᵣ[:,k-1] .= uᵣ
        all_msee_x[:,:,k] .= prob.msee_x₀
        all_goals_h[:,k] .= prob.goal
        all_goals_r[:,k] .= prob.robot_goal
        all_dist[k-1] = norm(xₕ[1:2]-xᵣ[1:2], 2)
        
        # compute and store metrics
        pred_err[k-1] = err
        θ_est_err[k-1] = norm(θ_true - prob.θ₀)
        msee_norm[k-1] = norm(prob.msee_x₀)
    end
    return Dict("all_xₕ"=>all_xₕ, "all_xᵣ"=>all_xᵣ, "all_uᵣ"=>all_uᵣ, "all_goals_h"=>all_goals_h, "all_goals_r"=>all_goals_r,
                "all_dist"=>all_dist, "all_goals"=>all_goals, "pred_err"=>pred_err, 
                "θ_est_err"=>θ_est_err, "msee_norm"=>msee_norm, "points"=>points, "all_times"=>all_times)
end