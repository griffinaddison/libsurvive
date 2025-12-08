#!/usr/bin/env python3
import ctypes
import math
import signal
import sys
import os

import numpy as np

import pysurvive






# --- ARX5 SDK hookup -------------------------------------------------
SDK_ROOT = "/home/griffin/arx5-sdk"
sys.path.append(f"{SDK_ROOT}/python")
os.environ["LD_LIBRARY_PATH"] = f"{SDK_ROOT}/lib/x86_64:" + os.environ.get("LD_LIBRARY_PATH", "")

from arx5_interface import Arx5CartesianController, EEFState, RobotConfigFactory, ControllerConfigFactory
# ---------------------------------------------------------------------











def _copy_pose(dst, src):
    for i in range(3):
        dst.Pos[i] = src.Pos[i]
    for i in range(4):
        dst.Rot[i] = src.Rot[i]


def _set_identity(pose):
    for i in range(3):
        pose.Pos[i] = 0.0
    pose.Rot[0] = 1.0
    pose.Rot[1] = 0.0
    pose.Rot[2] = 0.0
    pose.Rot[3] = 0.0


def _quat_cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _quat_rotate(q, v):
    qw, qx, qy, qz = q
    q_vec = (qx, qy, qz)
    uv = _quat_cross(q_vec, v)
    uuv = _quat_cross(q_vec, uv)
    uv = tuple(val * (2.0 * qw) for val in uv)
    uuv = tuple(val * 2.0 for val in uuv)
    return (
        v[0] + uv[0] + uuv[0],
        v[1] + uv[1] + uuv[1],
        v[2] + uv[2] + uuv[2],
    )


def _quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _invert_pose(pose):
    inv = pysurvive.SurvivePose()
    inv.Rot[0] = pose.Rot[0]
    inv.Rot[1] = -pose.Rot[1]
    inv.Rot[2] = -pose.Rot[2]
    inv.Rot[3] = -pose.Rot[3]
    rotated = _quat_rotate(
        (inv.Rot[0], inv.Rot[1], inv.Rot[2], inv.Rot[3]),
        (pose.Pos[0], pose.Pos[1], pose.Pos[2]),
    )
    for i in range(3):
        inv.Pos[i] = -rotated[i]
    return inv


def _apply_pose(lhs, rhs):
    out = pysurvive.SurvivePose()
    rotated = _quat_rotate(
        (lhs.Rot[0], lhs.Rot[1], lhs.Rot[2], lhs.Rot[3]),
        (rhs.Pos[0], rhs.Pos[1], rhs.Pos[2]),
    )
    for i in range(3):
        out.Pos[i] = rotated[i] + lhs.Pos[i]
    rot = _quat_multiply(
        (lhs.Rot[0], lhs.Rot[1], lhs.Rot[2], lhs.Rot[3]),
        (rhs.Rot[0], rhs.Rot[1], rhs.Rot[2], rhs.Rot[3]),
    )
    for i in range(4):
        out.Rot[i] = rot[i]
    return out


def _pose_from_event(event_pose):
    pose = pysurvive.SurvivePose()
    _copy_pose(pose, event_pose)
    return pose

def _quat_to_euler(quat):
    w, x, y, z = quat
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def _format_pose(pose):
    pos = [pose.Pos[i] for i in range(3)]
    rot = [pose.Rot[i] for i in range(4)]
    return (
        f"{pos[0]:6.2f} {pos[1]:6.2f} {pos[2]:6.2f}",
        f"{rot[0]:6.2f} {rot[1]:6.2f} {rot[2]:6.2f} {rot[3]:6.2f}",
    )


class ArmTeleopState:
    def __init__(self, name, controller):
        self.name = name
        self.controller = controller
        self.engaged = False
        self.have_reference = False
        self.reference_pose = pysurvive.SurvivePose()
        self.reference_pose_inv = pysurvive.SurvivePose()
        _set_identity(self.reference_pose)
        _set_identity(self.reference_pose_inv)
        self.eef_reference_pose = None
        self.target_pose_6d = controller.get_home_pose().copy()
        robot_config = controller.get_robot_config()
        self.gripper_width = robot_config.gripper_width
        self.gripper_min = 0.0
        self.gripper_command = self.gripper_width
        self.gripper_hold = False
        self.last_command_pose = controller.get_eef_state().pose_6d().copy()
        self.joint_torque_limit = np.array(robot_config.joint_torque_max) * 0.5
        self.torque_limited = False
        self.workspace_clamped = False
        self.velocity_limited = False


def _create_controller(label):
    robot_cfg = RobotConfigFactory.get_instance().get_config("L5")
    jt_max = np.array(robot_cfg.joint_torque_max, copy=True)
    robot_cfg.joint_torque_max = jt_max
    jv_max = np.array(robot_cfg.joint_vel_max, copy=True)
    print(f"[{label}] Original joint vel max:", jv_max)
    jv_max = [3, 3, 3, 4, 4, 4]
    robot_cfg.joint_vel_max = jv_max
    ctrl_cfg = ControllerConfigFactory.get_instance().get_config(
        "cartesian_controller", robot_cfg.joint_dof
    )
    return robot_cfg, ctrl_cfg


def _initialize_controller_state(controller):
    controller.reset_to_home()
    curr_pose = controller.get_eef_state().pose_6d().copy()
    curr_pose[:3] = np.array([0.243, 0.0, 0.317])
    robot_config = controller.get_robot_config()
    target = EEFState(curr_pose, robot_config.gripper_width)
    target.timestamp = controller.get_timestamp() + 2.0
    controller.set_eef_cmd(target)


def main():
    # --- init libsurvive context --------------------------------------
    ctx = pysurvive.SimpleContext(sys.argv)
    if not ctx.ptr:
        print("Failed to initialize libsurvive context", file=sys.stderr)
        sys.exit(1)

    for obj in ctx.Objects():
        print(obj.Name().decode("utf-8"))

    # --- init ARX5 controllers ---------------------------------------
    interfaces = {
        pysurvive.SURVIVE_OBJECT_SUBTYPE_KNUCKLES_L: ("left", "can0"),
        pysurvive.SURVIVE_OBJECT_SUBTYPE_KNUCKLES_R: ("right", "can1"),
    }

    arm_states = {}
    controllers = {}
    for subtype, (name, interface) in interfaces.items():
        robot_cfg, ctrl_cfg = _create_controller(name)
        controller = Arx5CartesianController(robot_cfg, ctrl_cfg, interface)
        _initialize_controller_state(controller)
        controllers[subtype] = controller
        arm_states[subtype] = ArmTeleopState(name, controller)
        gain = controller.get_gain()
        print(
            f"{name.capitalize()} initial gains:\n"
            f"  kp: {np.array2string(gain.kp(), precision=4)}\n"
            f"  kd: {np.array2string(gain.kd(), precision=4)}"
        )

    preview_time = 0.05        # seconds ahead of current time
    pos_scale = 1.25           # scale controller meters -> arm meters (tune)  ### NEW
    workspace_limit = 0.7    # +/- 70 cm relative to captured pose
    max_translation_step = 0.08  # limit per-update delta to 5 cm
    rot_scale = 1.25
    torque_limit = 0.80
    # -------------------------------------------------------------------

    keep_running = True

    def stop_handler(signum, frame):
        nonlocal keep_running
        if not keep_running:
            raise SystemExit(1)
        keep_running = False

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    event = pysurvive.SurviveSimpleEvent()
    try:
        while keep_running:
            event_type = pysurvive.survive_simple_wait_for_event(ctx.ptr, ctypes.byref(event))
            if event_type == pysurvive.SurviveSimpleEventType_Shutdown:
                break
            if event_type == pysurvive.SurviveSimpleEventType_None:
                continue

            if event_type == pysurvive.SurviveSimpleEventType_PoseUpdateEvent:
                pose_event = pysurvive.survive_simple_get_pose_updated_event(ctypes.byref(event))
                if not pose_event or not pose_event.contents.object:
                    continue
                obj = pose_event.contents.object
                subtype = pysurvive.survive_simple_object_get_subtype(obj)
                hand_state = arm_states.get(subtype)
                if not hand_state or not hand_state.engaged:
                    continue

                pose = _pose_from_event(pose_event.contents.pose)
                if hand_state.have_reference:
                    # pose is now "delta" in controller frame
                    pose = _apply_pose(hand_state.reference_pose_inv, pose)

                    # --- NEW: use delta pose to command ARX5 -----------------
                    if hand_state.eef_reference_pose is not None:
                        controller = hand_state.controller
                        joint_state = controller.get_joint_state()
                        joint_torques = joint_state.torque().copy()
                        abs_torque = np.abs(joint_torques)
                        over_limit = abs_torque - hand_state.joint_torque_limit
                        exceeding = np.where(over_limit > 0)[0]
                        if exceeding.size > 0:
                            idx = exceeding[np.argmax(over_limit[exceeding])]
                            actual_val = abs_torque[idx]
                            limit_val = hand_state.joint_torque_limit[idx]
                            if not hand_state.torque_limited:
                                print(
                                    f"{hand_state.name.capitalize()} joint torque limit exceeded "
                                    f"(joint {idx}: {actual_val:.4f} Nm > {limit_val:.4f} Nm); holding pose"
                                )
                                hand_state.torque_limited = True
                            continue
                        elif hand_state.torque_limited:
                            print(f"{hand_state.name.capitalize()} torque back within limits; resuming motion")
                            hand_state.torque_limited = False
                        if (
                            not hand_state.gripper_hold
                            and abs(joint_state.gripper_torque) > torque_limit
                        ):
                            hand_state.gripper_command = max(
                                hand_state.gripper_min,
                                min(hand_state.gripper_width, joint_state.gripper_pos),
                            )
                            hand_state.gripper_hold = True
                            print(
                                f"{hand_state.name.capitalize()} gripper torque {joint_state.gripper_torque:.3f} Nm exceeding limit; holding at {hand_state.gripper_command:.4f} m"
                            )

                        # simple position-only mapping L5-frame ~= world-frame
                        dx, dy, dz = pose.Pos[0], pose.Pos[1], pose.Pos[2]

                        hand_state.target_pose_6d[:] = hand_state.eef_reference_pose
                        hand_state.target_pose_6d[0] += -pos_scale * dy
                        hand_state.target_pose_6d[1] += -pos_scale * dx
                        hand_state.target_pose_6d[2] += -pos_scale * dz
                        workspace_clamped = False
                        workspace_reason = ""
                        for idx in range(3):
                            min_bound = hand_state.eef_reference_pose[idx] - workspace_limit
                            max_bound = hand_state.eef_reference_pose[idx] + workspace_limit
                            clamped_val = max(
                                min_bound, min(max_bound, hand_state.target_pose_6d[idx])
                            )
                            if clamped_val != hand_state.target_pose_6d[idx]:
                                if not workspace_clamped:
                                    requested_offset = hand_state.target_pose_6d[idx] - hand_state.eef_reference_pose[idx]
                                    workspace_reason = (
                                        f"axis {idx}: {requested_offset:+.4f} m exceeds ±{workspace_limit:.4f} m"
                                    )
                                workspace_clamped = True
                                hand_state.target_pose_6d[idx] = clamped_val
                        if workspace_clamped and not hand_state.workspace_clamped:
                            print(f"{hand_state.name.capitalize()} workspace clamp active ({workspace_reason})")
                        elif not workspace_clamped and hand_state.workspace_clamped:
                            print(f"{hand_state.name.capitalize()} workspace clamp cleared")
                        hand_state.workspace_clamped = workspace_clamped
                        if hand_state.last_command_pose is not None:
                            velocity_limited = False
                            velocity_reason = ""
                            for idx in range(3):
                                prev = hand_state.last_command_pose[idx]
                                delta = hand_state.target_pose_6d[idx] - prev
                                if abs(delta) > max_translation_step:
                                    hand_state.target_pose_6d[idx] = prev + math.copysign(
                                        max_translation_step, delta
                                    )
                                    velocity_limited = True
                                    if not velocity_reason:
                                        velocity_reason = (
                                            f"axis {idx}: step {abs(delta):.4f} m > {max_translation_step:.4f} m limit"
                                        )
                            if velocity_limited and not hand_state.velocity_limited:
                                print(f"{hand_state.name.capitalize()} translation slew limit active ({velocity_reason})")
                            elif not velocity_limited and hand_state.velocity_limited:
                                print(f"{hand_state.name.capitalize()} translation slew limit cleared")
                            hand_state.velocity_limited = velocity_limited
                        roll, pitch, yaw = _quat_to_euler(pose.Rot)
                        hand_state.target_pose_6d[3] = hand_state.eef_reference_pose[3] - rot_scale * pitch
                        hand_state.target_pose_6d[4] = hand_state.eef_reference_pose[4] - rot_scale * roll #intuitive pitch
                        hand_state.target_pose_6d[5] = hand_state.eef_reference_pose[5] - rot_scale * yaw

                        eef_cmd = EEFState()
                        eef_cmd.pose_6d()[:] = hand_state.target_pose_6d
                        eef_cmd.gripper_pos = hand_state.gripper_command
                        eef_cmd.timestamp = controller.get_timestamp() + preview_time
                        hand_state.controller.set_eef_cmd(eef_cmd)
                        hand_state.last_command_pose = hand_state.target_pose_6d.copy()
                    # ---------------------------------------------------------

                # still handy to print for debugging if you want
                # name = pysurvive.survive_simple_object_name(obj).decode("utf-8")
                # serial = pysurvive.survive_simple_serial_number(obj).decode("utf-8")
                # pos_str, rot_str = _format_pose(pose)
                # print(f"{name} {serial} ({pose_event.contents.time:7.3f}):")
                # print(f"  {pos_str}")
                # print(f"  {rot_str}")

            elif event_type == pysurvive.SurviveSimpleEventType_ButtonEvent:
                button_event = pysurvive.survive_simple_get_button_event(ctypes.byref(event))
                if not button_event or not button_event.contents.object:
                    continue
                obj = button_event.contents.object
                subtype = pysurvive.survive_simple_object_get_subtype(obj)
                hand_state = arm_states.get(subtype)
                if not hand_state:
                    continue

                for i in range(button_event.contents.axis_count):
                    axis_id = button_event.contents.axis_ids[i]
                    if axis_id == pysurvive.SURVIVE_AXIS_TRIGGER:
                        trigger_val = button_event.contents.axis_val[i]
                        trigger_val = max(0.0, min(1.0, trigger_val))
                        hand_state.gripper_command = hand_state.gripper_width * (1.0 - trigger_val)
                        if hand_state.gripper_command < hand_state.gripper_min:
                            hand_state.gripper_command = hand_state.gripper_min
                        hand_state.gripper_hold = False

                if button_event.contents.button_id == pysurvive.SURVIVE_BUTTON_A:
                    if button_event.contents.event_type == pysurvive.SURVIVE_INPUT_EVENT_BUTTON_DOWN:
                        hand_state.engaged = True
                        pysurvive.survive_simple_object_get_latest_pose(
                            obj, ctypes.byref(hand_state.reference_pose)
                        )
                        inv = _invert_pose(hand_state.reference_pose)
                        _copy_pose(hand_state.reference_pose_inv, inv)
                        hand_state.have_reference = True

                        # snapshot current EE pose as reference                  ### NEW
                        eef_state = hand_state.controller.get_eef_state()
                        hand_state.eef_reference_pose = eef_state.pose_6d().copy()
                        hand_state.last_command_pose = hand_state.eef_reference_pose.copy()
                        hand_state.torque_limited = False
                        print(f"{hand_state.name.capitalize()} A button engaged")

                    elif button_event.contents.event_type == pysurvive.SURVIVE_INPUT_EVENT_BUTTON_UP:
                        hand_state.engaged = False
                        hand_state.have_reference = False
                        hand_state.eef_reference_pose = None     # forget ref on release   ### NEW
                        hand_state.gripper_hold = False
                        hand_state.last_command_pose = hand_state.controller.get_eef_state().pose_6d().copy()
                        hand_state.torque_limited = False
                        _set_identity(hand_state.reference_pose)
                        _set_identity(hand_state.reference_pose_inv)
                        print(f"{hand_state.name.capitalize()} A button released")
                elif button_event.contents.button_id == pysurvive.SURVIVE_BUTTON_B:
                    if button_event.contents.event_type == pysurvive.SURVIVE_INPUT_EVENT_BUTTON_DOWN:
                        print(f"{hand_state.name.capitalize()} B button pressed — initiating shutdown")
                        stop_handler(signal.SIGINT, None)

            elif event_type == pysurvive.SurviveSimpleEventType_DeviceAdded:
                obj_event = pysurvive.survive_simple_get_object_event(ctypes.byref(event))
                if obj_event and obj_event.contents.object:
                    name = pysurvive.survive_simple_object_name(obj_event.contents.object).decode("utf-8")
                    print(f"({obj_event.contents.time:.3f}) Found '{name}'")
    finally:
        print("Cleaning up")
        pysurvive.survive_simple_close(ctx.ptr)
        for subtype, controller in controllers.items():
            hand_state = arm_states.get(subtype)
            label = hand_state.name.capitalize() if hand_state else "Arm"
            try:
                print(f"{label}: returning to home pose")
                controller.reset_to_home()
            except Exception as e:
                print(f"Error during shutdown for {label}: {e}")


if __name__ == "__main__":
    main()
