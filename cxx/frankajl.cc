#include <array>
#include <chrono>
#include <iostream>

#include "Eigen/Dense"

#include "franka/control_types.h"
#include "franka/model.h"
#include "franka/robot_state.h"
#include "franka/robot.h"

#include <julia.h>

#include "jlcxx/jlcxx.hpp"

#include "jlcxx/const_array.hpp"
#include "jlcxx/functions.hpp"
#include "jlcxx/stl.hpp"
#include "jlcxx/tuple.hpp"


namespace {
void _SetDefaultBehavior(franka::Robot *robot) {
  robot->setCollisionBehavior({{99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0}},
                              {{99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0}},
                              {{99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0}},
                              {{99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0}},
                              {{99.0, 99.0, 99.0, 99.0, 99.0, 99.0}},
                              {{99.0, 99.0, 99.0, 99.0, 99.0, 99.0}},
                              {{99.0, 99.0, 99.0, 99.0, 99.0, 99.0}},
                              {{99.0, 99.0, 99.0, 99.0, 99.0, 99.0}});
  robot->setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});
  constexpr double kXYZ = 3000.0;
  constexpr double kRPY = 300.0;
  robot->setCartesianImpedance({{kXYZ, kXYZ, kXYZ, kRPY, kRPY, kRPY}});
}

template <typename T> T Zeros();
template <> franka::JointPositions Zeros<franka::JointPositions>() {
  return franka::JointPositions({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
}
template <> franka::JointVelocities Zeros<franka::JointVelocities>() {
  return franka::JointVelocities({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
}
template <> franka::CartesianPose Zeros<franka::CartesianPose>() {
  return franka::CartesianPose({1.0, 0.0, 0.0, 0.0, //
                                0.0, 1.0, 0.0, 0.0, //
                                0.0, 0.0, 1.0, 0.0, //
                                0.0, 0.0, 0.0, 1.0});
}
template <> franka::CartesianVelocities Zeros<franka::CartesianVelocities>() {
  return franka::CartesianVelocities({0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
}
template <> franka::Torques Zeros<franka::Torques>() {
  return franka::Torques({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
}

template <typename T>
using FrankaCallback = void (*)(T *, void *, const franka::RobotState *,
                                const franka::Duration *);
} // namespace

struct RobotInterface {
  franka::Robot robot;

  RobotInterface(const std::string &ip) : robot(ip) {}

  void SetDefaultBehavior() { _SetDefaultBehavior(&robot); }
  void AutomaticErrorRecovery() { robot.automaticErrorRecovery(); }

  franka::RobotState ReadOnce() { return robot.readOnce(); }

  franka::Model LoadModel() { return robot.loadModel(); }

  template <typename T> void Control(void *data, FrankaCallback<T> controller) {
    auto controller_wrapper = [data,
                               controller](const franka::RobotState &state,
                                           franka::Duration duration) {
      T output = Zeros<T>();
      controller(&output, data, &state, &duration);
      return output;
    };

    controller_wrapper(robot.readOnce(), franka::Duration(0));
    robot.control(controller_wrapper);
  }

  template <typename T>
  void ControlTorques(void *controller_data,
                      FrankaCallback<franka::Torques> controller,
                      void *generator_data, FrankaCallback<T> generator) {
    auto controller_wrapper = [controller_data,
                               controller](const franka::RobotState &state,
                                           franka::Duration duration) {
      franka::Torques output = Zeros<franka::Torques>();
      controller(&output, controller_data, &state, &duration);
      return output;
    };
    auto generator_wrapper = [generator_data,
                              generator](const franka::RobotState &state,
                                         franka::Duration duration) {
      T output = Zeros<T>();
      generator(&output, generator_data, &state, &duration);
      return output;
    };

    controller_wrapper(robot.readOnce(), franka::Duration(0));
    generator_wrapper(robot.readOnce(), franka::Duration(0));
    robot.control(controller_wrapper, generator_wrapper);
  }
};

// Implementation note: This helps work around a double registration issue in
// CxxWrap.jl
#define NAMED_CONTROL(type)                                                    \
  method("_control_" #type, &RobotInterface::Control<franka::type>)
#define NAMED_CONTROL_GENERATOR(type)                                          \
  method("_control_Torques" #type,                                             \
         &RobotInterface::ControlTorques<franka::type>)

template <typename T, std::size_t N>
void add_array_type(jlcxx::Module &mod, const std::string &name) {
  auto super_type_generic = jlcxx::julia_type("AbstractVector");
  auto parameter_type = jlcxx::julia_type("Float64");
  auto super_type = jlcxx::apply_type((jl_value_t *)super_type_generic,
                                      (jl_value_t **)&parameter_type, 1);
  mod.add_type<std::array<T, N>>(name, super_type)
      .template constructor<>()
      .template constructor<const std::array<T, N> &>();
  mod.method("size",
             [](const std::array<T, N> &a) { return std::make_tuple(N); });
  mod.method("length", [](const std::array<T, N> &a) { return N; });
  mod.method("getindex", [](const std::array<T, N> &a, std::size_t i) {
    return (a)[i - 1];
  });
  mod.method("setindex!",
             [](std::array<T, N> &a, T v, std::size_t i) { (a)[i - 1] = v; });
  mod.method("firstindex", [](const std::array<T, N> &a) { return 1; });
  mod.method("lastindex", [](const std::array<T, N> &a) { return N; });
}

#define CONST_ARRAY_GETTER(type, name)                                         \
  mod.method("get_" #name, [](const type *v) {                                 \
    return jlcxx::make_const_array(v->name.data(), v->name.size());            \
  });                                                                          \
  mod.method("get_" #name, [](const type &v) {                                 \
    return jlcxx::make_const_array(v.name.data(), v.name.size());              \
  })

#define CONST_VALUE_GETTER(type, name)                                         \
  mod.method("get_" #name, [](const type *v) { return v->name; })

#define MUTABLE_ARRAY_GETTER(type, field)                                      \
  mod.method("get_" #field, [](const type *v) { return &v->field; });

#define NAMED_MUTABLE_ARRAY_GETTER(type, name, field)                          \
  mod.method(#name, [](const type *v) { return &v->field; });

#define EXPOSE_MOTION_FINISHED(type)                                           \
  mod.method("set_motion_finished!",                                           \
             [](type *v, bool value) { v->motion_finished = value; });         \
  mod.method("get_motion_finished",                                            \
             [](const type *v) { return v->motion_finished; });

namespace jlcxx {
template <> struct IsMirroredType<std::array<double, 2>> : std::false_type {};
template <> struct IsMirroredType<std::array<double, 3>> : std::false_type {};
template <> struct IsMirroredType<std::array<double, 6>> : std::false_type {};
template <> struct IsMirroredType<std::array<double, 7>> : std::false_type {};
template <> struct IsMirroredType<std::array<double, 9>> : std::false_type {};
template <> struct IsMirroredType<std::array<double, 16>> : std::false_type {};
template <> struct IsMirroredType<std::array<double, 42>> : std::false_type {};
template <> struct IsMirroredType<std::array<double, 49>> : std::false_type {};
} // namespace jlcxx

JLCXX_MODULE define_julia_module(jlcxx::Module &mod) {
  add_array_type<double, 2>(mod, "Array2d");
  add_array_type<double, 3>(mod, "Array3d");
  add_array_type<double, 6>(mod, "Array6d");
  add_array_type<double, 7>(mod, "Array7d");
  add_array_type<double, 9>(mod, "Array9d");
  add_array_type<double, 16>(mod, "Array16d");
  add_array_type<double, 42>(mod, "Array42d");
  add_array_type<double, 49>(mod, "Array49d");

  mod.add_type<franka::Duration>("Duration")
      .constructor<uint64_t>()
      .method("to_sec", &franka::Duration::toSec)
      .method("to_msec", &franka::Duration::toMSec);

  mod.add_type<franka::RobotState>("RobotState");
  CONST_ARRAY_GETTER(franka::RobotState, O_T_EE);
  CONST_ARRAY_GETTER(franka::RobotState, O_T_EE_d);
  CONST_ARRAY_GETTER(franka::RobotState, F_T_EE);
  CONST_ARRAY_GETTER(franka::RobotState, F_T_NE);
  CONST_ARRAY_GETTER(franka::RobotState, NE_T_EE);
  CONST_ARRAY_GETTER(franka::RobotState, EE_T_K);
  CONST_VALUE_GETTER(franka::RobotState, m_ee);
  CONST_ARRAY_GETTER(franka::RobotState, I_ee);
  CONST_ARRAY_GETTER(franka::RobotState, F_x_Cee);
  CONST_VALUE_GETTER(franka::RobotState, m_load);
  CONST_ARRAY_GETTER(franka::RobotState, I_load);
  CONST_ARRAY_GETTER(franka::RobotState, F_x_Cload);
  CONST_VALUE_GETTER(franka::RobotState, m_total);
  CONST_ARRAY_GETTER(franka::RobotState, I_total);
  CONST_ARRAY_GETTER(franka::RobotState, F_x_Ctotal);
  CONST_ARRAY_GETTER(franka::RobotState, elbow);
  CONST_ARRAY_GETTER(franka::RobotState, elbow_d);
  CONST_ARRAY_GETTER(franka::RobotState, elbow_c);
  CONST_ARRAY_GETTER(franka::RobotState, delbow_c);
  CONST_ARRAY_GETTER(franka::RobotState, ddelbow_c);
  CONST_ARRAY_GETTER(franka::RobotState, tau_J);
  CONST_ARRAY_GETTER(franka::RobotState, tau_J_d);
  CONST_ARRAY_GETTER(franka::RobotState, dtau_J);
  CONST_ARRAY_GETTER(franka::RobotState, q);
  CONST_ARRAY_GETTER(franka::RobotState, q_d);
  CONST_ARRAY_GETTER(franka::RobotState, dq);
  CONST_ARRAY_GETTER(franka::RobotState, dq_d);
  CONST_ARRAY_GETTER(franka::RobotState, ddq_d);
  CONST_ARRAY_GETTER(franka::RobotState, joint_contact);
  CONST_ARRAY_GETTER(franka::RobotState, cartesian_contact);
  CONST_ARRAY_GETTER(franka::RobotState, joint_collision);
  CONST_ARRAY_GETTER(franka::RobotState, cartesian_collision);
  CONST_ARRAY_GETTER(franka::RobotState, tau_ext_hat_filtered);
  CONST_ARRAY_GETTER(franka::RobotState, O_F_ext_hat_K);
  CONST_ARRAY_GETTER(franka::RobotState, K_F_ext_hat_K);
  CONST_ARRAY_GETTER(franka::RobotState, O_dP_EE_d);
  // CONST_ARRAY_GETTER(franka::RobotState, O_ddP_O);
  CONST_ARRAY_GETTER(franka::RobotState, O_T_EE_c);
  CONST_ARRAY_GETTER(franka::RobotState, O_dP_EE_c);
  CONST_ARRAY_GETTER(franka::RobotState, O_ddP_EE_c);
  CONST_ARRAY_GETTER(franka::RobotState, theta);
  CONST_ARRAY_GETTER(franka::RobotState, dtheta);
  // TODO: Errors current_errors;
  // TODO: Errors last_motion_errors;
  CONST_VALUE_GETTER(franka::RobotState, control_command_success_rate);
  // TODO: RobotMode robot_mode;
  CONST_VALUE_GETTER(franka::RobotState, time);

  mod.add_bits<franka::Frame>("Frame", jlcxx::julia_type("CppEnum"));
  mod.set_const("kJoint1", franka::Frame::kJoint1);
  mod.set_const("kJoint2", franka::Frame::kJoint2);
  mod.set_const("kJoint3", franka::Frame::kJoint3);
  mod.set_const("kJoint4", franka::Frame::kJoint4);
  mod.set_const("kJoint5", franka::Frame::kJoint5);
  mod.set_const("kJoint6", franka::Frame::kJoint6);
  mod.set_const("kJoint7", franka::Frame::kJoint7);
  mod.set_const("kFlange", franka::Frame::kFlange);
  mod.set_const("kEndEffector", franka::Frame::kEndEffector);
  mod.set_const("kStiffness", franka::Frame::kStiffness);

  mod.add_type<franka::Model>("Model");
  mod.method("get_pose", [](const franka::Model &model, franka::Frame frame,
                            const franka::RobotState *state) {
    return model.pose(frame, *state);
  });
  mod.method("get_pose", [](const franka::Model &model, franka::Frame frame,
                            const std::array<double, 7> *q,
                            const std::array<double, 16> *F_T_EE,
                            const std::array<double, 16> *EE_T_K) {
    return model.pose(frame, *q, *F_T_EE, *EE_T_K);
  });
  mod.method("get_body_jacobian",
             [](const franka::Model &model, franka::Frame frame,
                const franka::RobotState *state) {
               return model.bodyJacobian(frame, *state);
             });
  mod.method("get_body_jacobian",
             [](const franka::Model &model, franka::Frame frame,
                const std::array<double, 7> *q,
                const std::array<double, 16> *F_T_EE,
                const std::array<double, 16> *EE_T_K) {
               return model.bodyJacobian(frame, *q, *F_T_EE, *EE_T_K);
             });
  mod.method("get_zero_jacobian",
             [](const franka::Model &model, franka::Frame frame,
                const franka::RobotState *state) {
               return model.zeroJacobian(frame, *state);
             });
  mod.method("get_zero_jacobian",
             [](const franka::Model &model, franka::Frame frame,
                const std::array<double, 7> *q,
                const std::array<double, 16> *F_T_EE,
                const std::array<double, 16> *EE_T_K) {
               return model.zeroJacobian(frame, *q, *F_T_EE, *EE_T_K);
             });
  mod.method("get_mass",
             [](const franka::Model &model, const franka::RobotState *state) {
               return model.mass(*state);
             });
  mod.method("get_mass",
             [](const franka::Model &model, const std::array<double, 7> *q,
                const std::array<double, 9> *I_total, double m_total,
                const std::array<double, 3> *F_x_Ctotal) {
               return model.mass(*q, *I_total, m_total, *F_x_Ctotal);
             });
  mod.method("get_coriolis",
             [](const franka::Model &model, const franka::RobotState *state) {
               return model.coriolis(*state);
             });
  mod.method("get_coriolis",
             [](const franka::Model &model, const std::array<double, 7> *q,
                const std::array<double, 7> *dq,
                const std::array<double, 9> *I_total, double m_total,
                const std::array<double, 3> *F_x_Ctotal) {
               return model.coriolis(*q, *dq, *I_total, m_total, *F_x_Ctotal);
             });
  mod.method("get_gravity",
             [](const franka::Model &model, const franka::RobotState *state) {
               return model.gravity(*state);
             });
  mod.method("get_gravity",
             [](const franka::Model &model, const franka::RobotState *state,
                const std::array<double, 3> &gravity_earth) {
               return model.gravity(*state, gravity_earth);
             });
  mod.method("get_gravity",
             [](const franka::Model &model, const std::array<double, 7> *q,
                double m_total, const std::array<double, 3> *F_x_Ctotal) {
               return model.gravity(*q, m_total, *F_x_Ctotal);
             });
  mod.method("get_gravity",
             [](const franka::Model &model, const std::array<double, 7> *q,
                double m_total, const std::array<double, 3> *F_x_Ctotal,
                const std::array<double, 3> &gravity_earth) {
               return model.gravity(*q, m_total, *F_x_Ctotal, gravity_earth);
             });

  mod.add_type<franka::Torques>("Torques");
  EXPOSE_MOTION_FINISHED(franka::Torques);
  MUTABLE_ARRAY_GETTER(franka::Torques, tau_J);
  NAMED_MUTABLE_ARRAY_GETTER(franka::Torques, _get_finishable_data_generic,
                             tau_J);

  mod.add_type<franka::JointPositions>("JointPositions");
  EXPOSE_MOTION_FINISHED(franka::JointPositions);
  MUTABLE_ARRAY_GETTER(franka::JointPositions, q);
  NAMED_MUTABLE_ARRAY_GETTER(franka::JointPositions,
                             _get_finishable_data_generic, q);

  mod.add_type<franka::JointVelocities>("JointVelocities");
  EXPOSE_MOTION_FINISHED(franka::JointVelocities);
  MUTABLE_ARRAY_GETTER(franka::JointVelocities, dq);
  NAMED_MUTABLE_ARRAY_GETTER(franka::JointVelocities,
                             _get_finishable_data_generic, dq);

  mod.add_type<franka::CartesianPose>("CartesianPose");
  EXPOSE_MOTION_FINISHED(franka::CartesianPose);
  MUTABLE_ARRAY_GETTER(franka::CartesianPose, O_T_EE);
  // TODO: add elbow support
  NAMED_MUTABLE_ARRAY_GETTER(franka::CartesianPose,
                             _get_finishable_data_generic, O_T_EE);

  mod.add_type<franka::CartesianVelocities>("CartesianVelocities");
  EXPOSE_MOTION_FINISHED(franka::CartesianVelocities);
  MUTABLE_ARRAY_GETTER(franka::CartesianVelocities, O_dP_EE);
  // TODO: add elbow support
  NAMED_MUTABLE_ARRAY_GETTER(franka::CartesianVelocities,
                             _get_finishable_data_generic, O_dP_EE);

  mod.add_type<RobotInterface>("RobotInterface")
      .constructor<const std::string &>()
      .method("set_default_behavior!", &RobotInterface::SetDefaultBehavior)
      .method("automatic_error_recovery!",
              &RobotInterface::AutomaticErrorRecovery)
      .method("read_once", &RobotInterface::ReadOnce)
      .method("load_model", &RobotInterface::LoadModel)
      .NAMED_CONTROL(Torques)
      .NAMED_CONTROL(JointPositions)
      .NAMED_CONTROL_GENERATOR(JointPositions)
      .NAMED_CONTROL(JointVelocities)
      .NAMED_CONTROL_GENERATOR(JointVelocities)
      .NAMED_CONTROL(CartesianPose)
      .NAMED_CONTROL_GENERATOR(CartesianPose)
      .NAMED_CONTROL(CartesianVelocities)
      .NAMED_CONTROL_GENERATOR(CartesianVelocities);
}

#undef ARRAY_DATA_WRAPPER
#undef NAMED_CONTROL
#undef NAMED_CONTROL_GENERATOR
