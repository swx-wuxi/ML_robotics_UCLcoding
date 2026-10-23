"""Controllers, stop criteria and setup helpers for the Robotnik mobile base + Kinova arm reaching task."""
from dataclasses import dataclass

import numpy as np
import pinocchio as pin
from simulation_and_control import wrap_angle


def quaternion2bearing(q_w, q_x, q_y, q_z):
    """Convert quaternion orientation to planar bearing (yaw angle)."""
    quat = pin.Quaternion(q_w, q_x, q_y, q_z)
    quat.normalize()
    base_euler = pin.rpy.matrixToRpy(quat.toRotationMatrix())
    return base_euler[2]


@dataclass
class ArmControllerParams:
    """Tuning parameters of the arm reaching controller."""
    reach_limit: float                   # m, max EE distance from the shoulder the arm is asked to reach
    kp: float = 100.0                    # joint PD position gain
    kd: float = 30.0                     # joint PD velocity gain
    max_joint_vel: float = 1.0           # rad/s
    max_q_lead: float = 0.15             # rad, max distance of the joint reference from the measured angle
    max_torque: float = 30.0             # N*m
    cartesian_kp: float = 1.5
    max_cartesian_speed: float = 0.25    # m/s
    min_ik_damping: float = 0.01
    max_ik_damping: float = 0.10
    singularity_threshold: float = 0.15
    limit_margin: float = 0.05           # rad, joints this close to a limit are treated as saturated


def arm_controller(dyn_model, q_pin, q_arm, qd_arm, q_des_arm, q_hold_arm, target_pos, ee_pos,
                   ee_frame_id, shoulder_frame_id, arm_lower, arm_upper, dt, params, reach_active):
    """Cartesian reaching controller for the 7-joint arm.

    Differential IK (singularity-damped, joint-limit aware) generates the joint reference, which a
    gravity-compensated joint PD law tracks. Forward kinematics must already be up to date in
    dyn_model.pin_data for q_pin. When reach_active is False the arm holds q_hold_arm.

    Returns (torque_arm, q_des_arm, qd_des_arm).
    """
    if not reach_active:
        q_des_arm = q_hold_arm.copy()
        qd_des_arm = np.zeros(7)
    else:
        # End-effector Jacobian in the world-aligned frame; columns 6..12 are the arm joints
        pin.computeJointJacobians(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        J_lwa = pin.computeFrameJacobian(
            dyn_model.pin_model, dyn_model.pin_data, q_pin, ee_frame_id, pin.LOCAL_WORLD_ALIGNED
        )
        J_arm = J_lwa[:3, 6:13]

        # Increase damping near singular poses to keep joint velocities bounded.
        sigma_min = np.linalg.svd(J_arm, compute_uv=False)[-1]
        damping = np.interp(
            sigma_min,
            [0.0, params.singularity_threshold],
            [params.max_ik_damping, params.min_ik_damping],
        )

        # Proportional Cartesian velocity command towards the (reach-limited) target
        shoulder_pos = dyn_model.pin_data.oMf[shoulder_frame_id].translation
        offset = target_pos - shoulder_pos
        if np.linalg.norm(offset) > params.reach_limit:
            offset *= params.reach_limit / np.linalg.norm(offset)
        ik_error = shoulder_pos + offset - ee_pos
        v_cartesian = params.cartesian_kp * ik_error
        cartesian_speed = np.linalg.norm(v_cartesian)
        if cartesian_speed > params.max_cartesian_speed:
            v_cartesian *= params.max_cartesian_speed / cartesian_speed

        # Saturation-aware solve: joints at a limit that the motion would push further into are
        # frozen, and their share of the motion is redistributed to the remaining joints.
        has_limits = arm_lower < arm_upper
        free = np.ones(7, dtype=bool)
        qd_des_arm = np.zeros(7)
        for _ in range(7):
            J_free = J_arm[:, free]
            regularized_jacobian = J_free @ J_free.T + damping**2 * np.eye(3)
            qd_des_arm = np.zeros(7)
            qd_des_arm[free] = J_free.T @ np.linalg.solve(regularized_jacobian, v_cartesian)
            pushing_lower = has_limits & (q_arm <= arm_lower + params.limit_margin) & (qd_des_arm < 0)
            pushing_upper = has_limits & (q_arm >= arm_upper - params.limit_margin) & (qd_des_arm > 0)
            newly_blocked = (pushing_lower | pushing_upper) & free
            if not newly_blocked.any():
                break
            free &= ~newly_blocked
            qd_des_arm[:] = 0.0
        max_joint_speed = np.max(np.abs(qd_des_arm))
        if max_joint_speed > params.max_joint_vel:
            qd_des_arm *= params.max_joint_vel / max_joint_speed
        q_des_arm = q_des_arm + qd_des_arm * dt
        # Anti-windup: for unreachable targets the reference must not run away from the measured state
        q_des_arm = np.clip(q_des_arm, q_arm - params.max_q_lead, q_arm + params.max_q_lead)
        # Keep the reference inside the limits so PD overshoot does not hit the hard stop
        q_des_arm = np.where(
            has_limits,
            np.clip(q_des_arm, arm_lower + params.limit_margin, arm_upper - params.limit_margin),
            q_des_arm,
        )

    # Joint-space PD law with gravity compensation
    torque_arm = params.kp * (q_des_arm - q_arm) + params.kd * (qd_des_arm - qd_arm)
    tau_g = pin.computeGeneralizedGravity(dyn_model.pin_model, dyn_model.pin_data, q_pin)
    torque_arm = np.clip(torque_arm + tau_g[6:13], -params.max_torque, params.max_torque)
    return torque_arm, q_des_arm, qd_des_arm


@dataclass
class StopCriteria:
    """Thresholds and running state for the simulation stop conditions."""
    reach_tolerance: float                     # m
    max_steps: int                             # 0 disables the step cap
    min_time: float = 8.0                      # s before reached / stalled can trigger
    stall_window: float = 3.0                  # s without meaningful improvement of the EE error
    stall_improvement: float = 0.002           # m
    best_ee_error: float = np.inf
    best_ee_error_time: float = 0.0


def check_stop(criteria, current_time, time_step, ee_dist_error, base_navigation_done):
    """Evaluate the stop conditions for the current step.

    Returns a message describing why the simulation should stop, or None to continue.
    Stall tracking only runs once the base navigation is done (the arm is reaching).
    """
    if criteria.max_steps and current_time / time_step >= criteria.max_steps:
        return f"Reached step cap {criteria.max_steps} at t={current_time:.2f}s"

    if base_navigation_done:
        if ee_dist_error < criteria.best_ee_error - criteria.stall_improvement:
            criteria.best_ee_error = ee_dist_error
            criteria.best_ee_error_time = current_time
        elif (current_time - criteria.best_ee_error_time > criteria.stall_window
              and current_time > criteria.min_time):
            return (f"Target not reachable: closest pose reached at t={current_time:.2f}s "
                    f"(best EE error {criteria.best_ee_error:.4f} m, current {ee_dist_error:.4f} m)")

    if base_navigation_done and ee_dist_error < criteria.reach_tolerance and current_time > criteria.min_time:
        return f"Target reached within {criteria.reach_tolerance*1000:.0f}mm tolerance at t={current_time:.2f}s!"
    return None


@dataclass
class BaseControllerParams:
    """Tuning parameters of the mobile-base navigation controller."""
    nav_timeout: float             # s, max time allowed for the base to reach its goal pose
    kp_pos: float = 1.0
    kp_ori: float = 3.0
    max_lin_vel: float = 0.25      # m/s
    min_lin_vel: float = 0.05      # m/s, avoids stalling against wheel friction near the goal
    max_ang_vel: float = 1.0       # rad/s
    pos_tol: float = 0.08          # m
    ang_tol: float = 0.05          # rad
    wheel_radius: float = 0.11
    # Effective track width for skid-steer (calibrated in simulation: yaw rate ~0.34 rad/s per rad/s of wheel speed)
    wheel_base_width: float = 0.66


@dataclass
class BaseNavState:
    """Navigation phases: align (turn in place toward goal) -> drive -> orient (face target) -> done."""
    phase: str = "align"
    done: bool = False


def base_controller(state, base_pos, base_bearing, goal_pos, goal_bearing, current_time, params):
    """Skid-steer base navigation to a goal pose; updates state in place.

    Returns the wheel velocity commands [front_right, front_left, back_left, back_right] (rad/s).
    Once navigation is done the base holds position (zero wheel velocities).
    """
    if state.done:
        return np.zeros(4)

    dist_to_goal = np.linalg.norm(base_pos[:2] - goal_pos)
    to_goal = goal_pos - base_pos[:2]
    heading_to_goal = np.arctan2(to_goal[1], to_goal[0])
    v_cmd, w_cmd = 0.0, 0.0
    if state.phase == "align":
        err = wrap_angle(heading_to_goal - base_bearing)
        if dist_to_goal < params.pos_tol or abs(err) < params.ang_tol:
            state.phase = "drive" if dist_to_goal >= params.pos_tol else "orient"
        else:
            w_cmd = np.clip(params.kp_ori * err, -params.max_ang_vel, params.max_ang_vel)
    if state.phase == "drive":
        err = wrap_angle(heading_to_goal - base_bearing)
        passed_goal = dist_to_goal < 0.15 and np.cos(err) < 0.0
        if dist_to_goal < params.pos_tol or passed_goal:
            state.phase = "orient"
        else:
            # Gentle steering while driving: skid-steer slips a lot when turning hard, so keep it mostly straight
            v_cmd = np.clip(params.kp_pos * dist_to_goal, params.min_lin_vel, params.max_lin_vel) * max(0.0, np.cos(err))
            w_cmd = np.clip(1.5 * err, -0.3, 0.3)
    if state.phase == "orient":
        err = wrap_angle(goal_bearing - base_bearing)
        if abs(err) < params.ang_tol:
            state.done = True
        else:
            w_cmd = np.clip(params.kp_ori * err, -params.max_ang_vel, params.max_ang_vel)
    lw = (v_cmd - w_cmd * params.wheel_base_width / 2) / params.wheel_radius
    rw = (v_cmd + w_cmd * params.wheel_base_width / 2) / params.wheel_radius
    if current_time > params.nav_timeout:
        state.done = True
    return np.array([rw, lw, lw, rw])


FINGER_JOINT_NAMES = [
    "robot_j2s7s300_joint_finger_1",
    "robot_j2s7s300_joint_finger_tip_1",
    "robot_j2s7s300_joint_finger_2",
    "robot_j2s7s300_joint_finger_tip_2",
    "robot_j2s7s300_joint_finger_3",
    "robot_j2s7s300_joint_finger_tip_3",
]


def configure_gripper_fingers(sim, client, rid, fixed_angles=None):
    """Reset the gripper fingers to fixed angles and hold them there.

    Intercepts the motor torque commands so wheels (0..3) and arm (4..10) keep their
    normal torque control while fingers (11..16) are position-held.
    Returns the finger target angles (initial config angles if `fixed_angles` is None).
    """
    finger_joint_ids = [
        j for j in range(client.getNumJoints(rid))
        if client.getJointInfo(rid, j)[1].decode() in FINGER_JOINT_NAMES
    ]
    raw_init_angles = np.array(sim.GetInitMotorAngles())
    target_finger_angles = fixed_angles if fixed_angles is not None else raw_init_angles[11:17]
    for i, jid in enumerate(finger_joint_ids):
        client.resetJointState(rid, jid, targetValue=float(target_finger_angles[i]), targetVelocity=0.0)
    sim.ReceiveObservation()

    orig_set_torque = sim._SetMotorTorqueByIds

    def custom_set_torque(motor_ids, commands, index=0):
        orig_set_torque(motor_ids[:11], commands[:11], index=index)
        sim.pybullet_client.setJointMotorControlArray(
            bodyIndex=sim.bot[index].bot_pybullet,
            jointIndices=motor_ids[11:17],
            controlMode=sim.pybullet_client.POSITION_CONTROL,
            targetPositions=np.asarray(target_finger_angles).tolist(),
            forces=[5.0] * 6
        )

    sim._SetMotorTorqueByIds = custom_set_torque
    return target_finger_angles
