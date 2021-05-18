using COSMO
include("SEA.jl")
include("BeliefCVX.jl")

function fill_Q2!(prob, Q)
    nᵣ, mᵣ, nₕ, mₕ = prob.sizes
    Σ_θ = prob.msee_θ₀
    Σinds = prob.Σ_inds
    for i=1:nₕ
        for j=1:nₕ
            Σ_cij = view(Σ_θ, Σinds[i], Σinds[j])
            Q .= Q + Σ_cij 
        end
    end
    Q .= Symmetric(Q)
end

function c_robot_dyn(prob)
    nᵣ, mᵣ, nₕ, mₕ = prob.sizes
    xin = prob.Z_inds[1]
    uin = prob.Z_inds[3]
    # extract states from times 1 to h-1
    A1 = zeros(nᵣ*(prob.h-1), prob.Z_inds[2][end][end])
    idx = 1
    for xi in xin[1:end-1]
        for i in xi
            A1[idx, i] = 1.0
            idx += 1
        end
    end
    
    # extract states from times 2 to h
    A2 = zeros(nᵣ*(prob.h-1), prob.Z_inds[2][end][end])
    idx = 1
    for xi in xin[2:end]
        for i in xi
            A2[idx, i] = 1.0
            idx += 1
        end
    end
    
    # extract controls
    B = zeros(mᵣ*(prob.h-1), prob.Z_inds[2][end][end])
    idx = 1
    for ui in uin
        for i in ui
            B[idx, i] = 1.0
            idx += 1
        end
    end
    
    # create dynamics matrix
    A_dyn = zeros(nᵣ*(prob.h-1), nᵣ*(prob.h-1))
    B_dyn = zeros(nᵣ*(prob.h-1), mᵣ*(prob.h-1))
    
    for i=1:(prob.h-1)
        xi = (i-1)*nᵣ .+ (1:nᵣ)
        ui = (i-1)*mᵣ .+ (1:mᵣ)
        A_dyn[xi, xi] .= prob.robot.A
        B_dyn[xi, ui] .= prob.robot.B
    end
    return A1, A2, B, A_dyn, B_dyn
end

function c_φ(prob)
    nᵣ, mᵣ, nₕ, mₕ = prob.sizes
    xᵣin = prob.Z_inds[1]
    xin = prob.Z_inds[2]
    A1 = zeros(nₕ+mₕ, prob.Z_inds[2][end][end])
    idx = 1
    for i in xin[end]
        A1[idx, i] = 1.0
        idx += 1
    end
    for i in xᵣin[end]
        A1[idx, i] = 1.0
        idx += 1
    end
    for i in xin[end]
        A1[idx, i] = 1.0
        idx += 1
    end
    
    A2 = zeros(nₕ+mₕ, prob.Z_inds[2][end][end])
    b = zeros(nₕ+mₕ)
    idx = nₕ+1
    for i in xin[end]
        A2[idx, i] = 1.0
        idx += 1
    end
    b[2nₕ+1:end] .= -prob.goal
    
    return A1 - A2, b
end

function c_human_dyn(prob)
    nᵣ, mᵣ, nₕ, mₕ = prob.sizes
    xᵣin = prob.Z_inds[1]
    xin = prob.Z_inds[2]
    
    # extract states from times 1 to h-1
    A1 = zeros(nₕ*(prob.h-1), prob.Z_inds[2][end][end])
    idx = 1
    for xi in xin[1:end-1]
        for i in xi
            A1[idx, i] = 1.0
            idx += 1
        end
    end
    
    # extract states from times 2 to h
    A2 = zeros(nₕ*(prob.h-1), prob.Z_inds[2][end][end])
    idx = 1
    for xi in xin[2:end]
        for i in xi
            A2[idx, i] = 1.0
            idx += 1
        end
    end
    
    # extract controls 
    B = zeros(mₕ*(prob.h-1), prob.Z_inds[2][end][end])
    uin = [(k-1)*mₕ .+ (1:mₕ) for k=1:prob.h-1]
    idx = 1
    b = zeros(mₕ*(prob.h-1))
    for ui in uin
        xᵣi = xᵣin[idx]
        xi = xin[idx]
        idx2 = 1
        for i in ui
            if idx2 <= 4
                B[i, xᵣi[idx2]] = 1.0
                B[i, xi[idx2]] = -1.0
            else
                B[i, xi[idx2-4]] = 1.0
                b[i] = prob.goal[idx2-4]
            end
            idx2 += 1
        end
        idx += 1
    end
    
    # human dynamics matrices 
    C = reshape(prob.θ₀, nₕ+mₕ, nₕ)'
    Aₕ = C[1:nₕ,1:nₕ]
    Bₕ = C[1:nₕ,nₕ+1:end]
    
    # create dynamics matrix
    A_dyn = zeros(nₕ*(prob.h-1), nₕ*(prob.h-1))
    B_dyn = zeros(nₕ*(prob.h-1), mₕ*(prob.h-1))
    
    for i=1:(prob.h-1)
        xi = (i-1)*nₕ .+ (1:nₕ)
        ui = (i-1)*mₕ .+ (1:mₕ)
        A_dyn[xi, xi] .= Aₕ
        B_dyn[xi, ui] .= Bₕ
    end
    
    return A1, A2, B, b, A_dyn, B_dyn
end

function c_torque(prob)
    nᵣ, mᵣ, nₕ, mₕ = prob.sizes
    uin = prob.Z_inds[3]
    
    # extract controls
    B = zeros(mᵣ*(prob.h-1), prob.Z_inds[2][end][end])
    idx = 1
    for ui in uin
        for i in ui
            B[idx, i] = 1.0
            idx += 1
        end
    end
    lb = zeros(mᵣ*(prob.h-1))
    lb .= prob.umin
    
    ub = zeros(mᵣ*(prob.h-1))
    ub .= prob.umax
    
    return B, lb, ub
end

function c_initial_pos(prob)
    nᵣ, mᵣ, nₕ, mₕ = prob.sizes
    A = zeros(nᵣ + nₕ, prob.Z_inds[2][end][end])
    A[1:nᵣ,1:nᵣ] .= I(nᵣ)
    A[nᵣ+1:nᵣ+nₕ,nᵣ+1:nᵣ+nₕ] .= I(nₕ)
    
    b = [prob.x₀; prob.x̂₀]
    
    return A, b
end

function c_final_pos(prob)
    nᵣ, mᵣ, nₕ, mₕ = prob.sizes
    A = zeros(nᵣ, prob.Z_inds[2][end][end])
    A[1:nᵣ,prob.Z_inds[1][end]] .= I(nᵣ)
    
    b = copy(prob.xf)
    return A, b
end

function c_dist(prob)
    nᵣ, mᵣ, nₕ, mₕ = prob.sizes
    xᵣin = prob.Z_inds[1]
    xin = prob.Z_inds[2]
    
    A_dist = [zeros(2, prob.Z_inds[2][end][end]) for k=1:prob.h-1]
    for k=1:prob.h-1
        idx = 1
        # extract states from times 2 to h
        Aₕ = zeros(2, prob.Z_inds[2][end][end])
        xi = xin[k+1]
        for i in xi[1:2]
            Aₕ[idx, i] = 1.0
            idx += 1
        end

        # extract states from times 2 to h
        Aᵣ = zeros(2, prob.Z_inds[2][end][end])
        idx = 1
        xi = xᵣin[k+1]
        for i in xi[1:2]
            Aᵣ[idx, i] = 1.0
            idx += 1
        end
        A_dist[k] .=  Aₕ - Aᵣ
    end

    lb = prob.dmin * ones(prob.h-1)
    ub = prob.dmax * ones(prob.h-1)
    return A_dist, lb, ub
end

function setup_COSMO(prob)
    n = prob.Z_inds[2][end][end]
    n_φ = prob.nₕ + prob.mₕ
    n_τ = prob.h-1
    Q = zeros(n_φ, n_φ)
    fill_Q2!(prob, Q)
    Q_full = zeros(n+n_φ, n+n_φ)
    Q_full[n+1:end,n+1:end] .= Q
    
    # setting up constraints
    # robot dynamics
    A1, A2, B, A_dyn, B_dyn = c_robot_dyn(prob)
    A_rob = A2 - A_dyn*A1 - B_dyn*B
    A_rob = [A_rob zeros(size(A_rob)[1], n_φ)]
    # human dynamics
    A1, A2, B, b2, A_dyn, B_dyn = c_human_dyn(prob)
    A_hum = A2 - A_dyn*A1 - B_dyn*B
    b_hum = B_dyn*b2
    A_hum = [A_hum zeros(size(A_hum)[1], n_φ)]
    # actuator limits
    B_torque, lb_torque, ub_torque = c_torque(prob)
    B_torque = [B_torque zeros(size(B_torque)[1], n_φ)]
    # initial position constraints
    A_init, b_init = c_initial_pos(prob)
    A_init = [A_init zeros(size(A_init)[1], n_φ)]
    # final position constraints
    A_final, b_final = c_final_pos(prob)
    A_final = [A_final zeros(size(A_final)[1], n_φ)]
    
    # φ constraints
    A_φ, b_φ = c_φ(prob)
    A_φ2 = [zeros(size(A_φ)) I(n_φ)]
    A_φ = [A_φ zeros(n_φ, n_φ)]
    A_φ = A_φ - A_φ2

    c_rob_dyn = COSMO.Constraint(A_rob, zeros(size(A_rob)[1]), COSMO.ZeroSet)
    c_hum_dyn = COSMO.Constraint(A_hum, b_hum, COSMO.ZeroSet)
    c_init = COSMO.Constraint(A_init, -b_init, COSMO.ZeroSet)
    c_final = COSMO.Constraint(A_final, -b_final, COSMO.ZeroSet)
    c_torque_lb = COSMO.Constraint(B_torque, -lb_torque, COSMO.Nonnegatives)
    c_torque_ub = COSMO.Constraint(-B_torque, ub_torque, COSMO.Nonnegatives)
    cons_φ = COSMO.Constraint(A_φ, b_φ, COSMO.ZeroSet)
    constraints = [c_rob_dyn; c_hum_dyn; c_init; c_torque_lb; c_torque_ub; cons_φ]
    
    model = COSMO.Model()
    COSMO.assemble!(model, Q_full, zeros(n+n_φ), constraints)
    
    return model, constraints
end

function switching_control(prob, k; σ_thresh = 0.1)
    # if norm(prob.msee_x₀, 2) <= σ_thresh
    if k > 100
        return get_uref(prob), 0
    else
        model, constraints = setup_COSMO(prob)
        results = COSMO.optimize!(model)
        uin = prob.Z_inds[3]
        uᵣ = results.x[uin[1]]
        
        return uᵣ, 1
    end
end

function set_xf(prob)
    xf = copy(prob.x₀)
    for k=1:prob.h-1
        uref = get_uref(prob)
        xf = prob.robot.A*xf + prob.robot.B*uref
    end
    prob.xf .= xf
end

function run_trial(;T=400, h=20, σ=1, cont=:active, ssa=true)
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
    n = prob.Z_inds[2][end][end]
    n_φ = prob.nₕ + prob.mₕ
    goals = [[20 .*rand(2) .- 10; 0; 0] for k=1:3]
    points = [0, 0]
    prob.robot_goal .= [(20 .*rand(2) .- 10); 0; 0]

    # storing data
    all_xₕ = [xₕ zeros(prob.nₕ, T-1)]
    all_xᵣ = [xᵣ zeros(prob.nᵣ, T-1)]
    all_uᵣ = zeros(prob.mᵣ, T-1)
    all_msee_x = zeros(prob.nₕ, prob.nₕ, T)
    all_msee_x[:,:,1] .= prob.msee_x₀
    all_goals_h = [prob.goal zeros(prob.nₕ, T-1)]
    all_goals_r = [prob.robot_goal zeros(prob.nₕ, T-1)]
    all_dist = zeros(T-1)
    controller = zeros(T-1)
    all_goals = []
    all_times = []

    # saving metrics
    pred_err = zeros(T-1)
    θ_est_err = zeros(T-1)
    msee_norm = zeros(T-1)
    noise = MvNormal(zeros(4), 0.1I)
    for k=2:T
        set_xf(prob)
        if cont == :active
            r = @timed switching_control(prob, k; σ_thresh=σ)
            uᵣ, c = r[1]
            push!(all_times, r[2])
        elseif cont == :sea
            uᵣ = get_uref(prob)
            c = 0
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
        controller[k-1] = c
        push!(all_goals, copy(goals))
        
        # compute and store metrics
        pred_err[k-1] = err
        θ_est_err[k-1] = norm(θ_true - prob.θ₀)
        msee_norm[k-1] = norm(prob.msee_x₀)
    end
    return Dict("all_xₕ"=>all_xₕ, "all_xᵣ"=>all_xᵣ, "all_uᵣ"=>all_uᵣ, "all_goals_h"=>all_goals_h, "all_goals_r"=>all_goals_r,
                "all_dist"=>all_dist, "controller"=>controller, "all_goals"=>all_goals, "pred_err"=>pred_err, 
                "θ_est_err"=>θ_est_err, "msee_norm"=>msee_norm, "points"=>points, "all_times"=>all_times)
end
