using Franka
using LinearAlgebra
using StaticArrays
using Rotations

Base.@kwdef mutable struct ZSineGenerator <: AbstractRobotController{CartesianPose}
    time::Float64 = 0.0
    center::Matrix{Float64} = zeros(4, 4)
    zA::Float64
    zf::Float64
end

function (c::ZSineGenerator)(state, dt::Real)
    c.time += dt
    finished = false
    if c.time == 0.0
        c.center = deepcopy(reshape(get_O_T_EE_c(state), 4, 4))
    end
    if c.time > 60.0
        finished = true
    end
    O_T_EE_c = deepcopy(c.center)
    Δz = c.zA * (cos(2π * c.zf * c.time) - 1)
    O_T_EE_c[3, 4] += Δz
    return (O_T_EE_c, finished)
end

Base.@kwdef mutable struct CartesianImpedanceController{
    ModelT<:Model,
} <: AbstractRobotController{Torques}
    time::Float64 = 0.0
    model::ModelT
    Kp::SMatrix{6,6,Float64,36}
    Kd::SMatrix{6,6,Float64,36}
end

function (c::CartesianImpedanceController)(stateptr, dt::Real)
    state = stateptr[]
    c.time += dt
    finished = c.time > 60.0

    goal = SMatrix{4,4,Float64}(get_O_T_EE_d(state))
    pose = SMatrix{4,4,Float64}(get_O_T_EE(state))
    Js = SMatrix{6,7,Float64}(get_zero_jacobian(c.model, kEndEffector, state))
    dq = SVector{7,Float64}(get_dq(state))

    coriolis = SVector{7,Float64}(get_coriolis(c.model, state))

    e_translation = pose[SOneTo(3), 4] - goal[SOneTo(3), 4]

    R_goal = QuatRotation(goal[1:3, 1:3])
    R_current = QuatRotation(pose[1:3, 1:3])
    quatdot(q1, q2) = q1.s * q2.s + q1.v1 * q2.v1 + q1.v2 * q2.v2 + q1.v3 * q2.v3
    if quatdot(R_goal.q, R_current.q) < 0
        R_current = QuatRotation(-R_current.q.s, -R_current.q.v1, -R_current.q.v2, -R_current.q.v3)
    end
    e_quat = QuatRotation(inv(R_current) * R_goal)
    e_rotation = -R_current * SA[e_quat.q.v1, e_quat.q.v2, e_quat.q.v3]

    error = SA[e_translation..., e_rotation...]

    Fs = -c.Kp * error - c.Kd * Js * dq
    tau_J = Js' * Fs + coriolis
    return (tau_J, finished)
end

function main()
    robot = Robot("172.16.0.2")
    automatic_error_recovery!(robot)

    torque_thresh = Franka.StlArray7d()
    torque_thresh .= 99
    force_thresh = Franka.StlArray6d()
    force_thresh .= 500
    set_collision_behavior!(robot, torque_thresh, torque_thresh, force_thresh, force_thresh)

    panda_home = [0.0, -π / 4, 0.0, -3π / 4, 0.0, π / 2, π / 4]
    control!(robot, JointGoalMotionGenerator(0.5, panda_home))

    Kp_translation = 5000.0
    Kp_rotation = 30.0
    Kp = Diagonal([Kp_translation, 0, Kp_translation, Kp_rotation, Kp_rotation, Kp_rotation])
    Kd = 2.0 * sqrt.(Kp)
    Kp = SMatrix{6,6,Float64}(Kp)
    Kd = SMatrix{6,6,Float64}(Kd)
    torque_controller = CartesianImpedanceController(; model=load_model(robot), Kp, Kd)

    generator = ZSineGenerator(; zA=0.1, zf=0.5)

    control!(robot, torque_controller, generator)
end

if !isinteractive()
    main()
end
