module Franka

using CxxWrap
using Libdl

# For std::array wrappers
import Base.setindex!
import Base.size
import Base.length
import Base.getindex
import Base.firstindex
import Base.lastindex

function __init__()
    @initcxx
end
@wrapmodule(() -> joinpath(@__DIR__, "../cxx/build/lib/libfrankajl.so"), :define_julia_module)

export AbstractRobotController
abstract type AbstractRobotController{OutputT} end

# Implementation note: @cfunctions need to be global
function _controller_wrapper!(
    output::CxxPtr{OutputT},
    bounced_controller_ptr::Ptr{Cvoid},
    state::ConstCxxPtr{RobotState},
    period::ConstCxxPtr{Duration}
)::Nothing where {OutputT}
    # Implementation note: @cfunctions need to be function pointers (i.e. only
    # into .text sections), so we pass the callable struct in as a pointer, pass
    # it back to ourselves in the callback wrapper on the other side, and unwrap
    # it here.
    bounced_controller = unsafe_pointer_to_objref(bounced_controller_ptr)
    dt = to_sec(period)

    # Call controller
    controller_output, motion_finished = bounced_controller(state, dt)
    @assert length(controller_output) == length(_get_finishable_data_generic(output))

    set_motion_finished!(output, motion_finished)
    for i in 1:length(controller_output)
        _get_finishable_data_generic(output)[i] = controller_output[i]
    end

    return nothing
end

function control!(
    robot::RobotInterface,
    controller::AbstractRobotController{OutputT},
) where {OutputT}
    c_controller_function_ptr = @safe_cfunction(
        _controller_wrapper!,
        Cvoid,
        (
            CxxPtr{OutputT},
            Ptr{Cvoid},
            ConstCxxPtr{RobotState},
            ConstCxxPtr{Duration}
        )
    )
    c_controller_data = pointer_from_objref(controller)

    control_driver = getfield(FrankaCommands, Symbol("_control_$OutputT"))
    GC.@preserve controller begin
        control_driver(robot, c_controller_data, c_controller_function_ptr)
    end
end

function control!(
    robot::RobotInterface,
    controller::AbstractRobotController{Torques},
    generator::AbstractRobotController{OutputT}
) where {OutputT}
    c_controller_function_ptr = @safe_cfunction(
        _controller_wrapper!,
        Cvoid,
        (
            CxxPtr{Torques},
            Ptr{Cvoid},
            ConstCxxPtr{RobotState},
            ConstCxxPtr{Duration}
        )
    )

    c_generator_function_ptr = @safe_cfunction(
        _controller_wrapper!,
        Cvoid,
        (
            CxxPtr{OutputT},
            Ptr{Cvoid},
            ConstCxxPtr{RobotState},
            ConstCxxPtr{Duration}
        )
    )

    c_controller_data = pointer_from_objref(controller)
    c_generator_data = pointer_from_objref(generator)

    control_driver = getfield(FrankaCommands, Symbol("_control_Torques$OutputT"))
    GC.@preserve controller generator begin
        control_driver(
            robot,
            c_controller_data,
            c_controller_function_ptr,
            c_generator_data,
            c_generator_function_ptr
        )
    end
end

# Duration
export Duration
export to_sec
export to_msec

# RobotState
export RobotState
export get_O_T_EE
export get_O_T_EE_d
export get_F_T_EE
export get_F_T_NE
export get_NE_T_EE
export get_EE_T_K
export get_m_ee
export get_I_ee
export get_F_x_Cee
export get_m_load
export get_I_load
export get_F_x_Cload
export get_m_total
export get_I_total
export get_F_x_Ctotal
export get_elbow
export get_elbow_d
export get_elbow_c
export get_delbow_c
export get_ddelbow_c
export get_tau_J
export get_tau_J_d
export get_dtau_J
export get_q
export get_q_d
export get_dq
export get_dq_d
export get_ddq_d
export get_joint_contact
export get_cartesian_contact
export get_joint_collision
export get_cartesian_collision
export get_tau_ext_hat_filtered
export get_O_F_ext_hat_K
export get_K_F_ext_hat_K
export get_O_dP_EE_d
# export get_O_ddP_O
export get_O_T_EE_c
export get_O_dP_EE_c
export get_O_ddP_EE_c
export get_theta
export get_dtheta
# TODO: Errors current_errors;
# TODO: Errors last_motion_errors;
export get_control_command_success_rate
# TODO: RobotMode robot_mode;
export get_time


# Model
export Frame
export kJoint1
export kJoint2
export kJoint3
export kJoint4
export kJoint5
export kJoint6
export kJoint7
export kFlange
export kEndEffector
export kStiffness

export Model
export get_pose
export get_body_jacobian
export get_zero_jacobian
export get_mass
export get_coriolis
export get_gravity

# Finishable (and associated)
export set_motion_finished!
export get_motion_finished

export Torques
export get_tau_J

export JointPositions
export get_q

export JointVelocities
export get_dq

export CartesianPose
export get_O_T_EE

export CartesianVelocities
export get_O_dP_EE

# RobotInterface
export RobotInterface
export set_default_behavior!
export automatic_error_recovery!
export control!
export read_once
export load_model

# Generators
include("./generators.jl")
export JointGoalMotionGenerator

end
