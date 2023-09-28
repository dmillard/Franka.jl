using Franka
using LinearAlgebra
using StaticArrays
using Rotations
using ForwardDiff

Base.@kwdef mutable struct CartesianConstraintController{
    ModelT<:Model,
} <: AbstractRobotController{Torques}
    time::Float64 = 0.0
    model::ModelT
    Kp::SMatrix{6,6,Float64,36}
    Kd::SMatrix{6,6,Float64,36}
    f::Function
end

function (c::CartesianConstraintController)(state, dt::Real)
    c.time += dt
    finished = c.time > 60.0

    goal = SMatrix{4,4,Float64}(get_O_T_EE_d(state))
    pose = SMatrix{4,4,Float64}(get_O_T_EE(state))
    Js = SMatrix{6,7,Float64}(get_zero_jacobian(c.model, kEndEffector, state))
    dq = SVector{7,Float64}(get_dq(state))

    coriolis = SVector{7,Float64}(get_coriolis(c.model, state))

    xyz = pose[SOneTo(3), 4]
    e_translation = c.f(xyz) * ForwardDiff.gradient(c.f, xyz)

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
    robot = RobotInterface("172.16.0.2")
    automatic_error_recovery!(robot)
    set_default_behavior!(robot)

    panda_home = [0.0, -π / 4, 0.0, -3π / 4, 0.0, π / 2, π / 4]
    control!(robot, JointGoalMotionGenerator(0.5, panda_home))

    EE = deepcopy(reshape(get_O_T_EE(read_once(robot)), 4, 4))

    function f(xyz)
        x, y, z = xyz - SA[0.306, 0.0, 0.59]
        r = 0.1

        #(z - 0.1 * (cos(20 * y) - 1))^2 + x^2
        return norm([x, y, z - r]) - r
    end

    Kp_translation = 5000.0
    Kp_rotation = 30.0
    Kp = Diagonal([Kp_translation, Kp_translation, Kp_translation, Kp_rotation, Kp_rotation, Kp_rotation])
    Kd = 2.0 * sqrt.(Kp)
    Kp = SMatrix{6,6,Float64}(Kp)
    Kd = SMatrix{6,6,Float64}(Kd)

    torque_controller = CartesianConstraintController(; model=load_model(robot), Kp, Kd, f)

    control!(robot, torque_controller)
end

if !isinteractive()
    main()
end
