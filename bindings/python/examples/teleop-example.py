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

import arx5_interface as arx5  # this is the pybind module
from arx5_interface import Arx5CartesianController, EEFState          # convenience
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


def main():
    # --- init libsurvive context --------------------------------------
    ctx = pysurvive.SimpleContext(sys.argv)
    if not ctx.ptr:
        print("Failed to initialize libsurvive context", file=sys.stderr)
        sys.exit(1)

    for obj in ctx.Objects():
        print(obj.Name().decode("utf-8"))

    # --- init ARX5 controller (L5 on can0) ----------------------------  ### NEW
    model = "L5"
    interface = "can1"
    controller = Arx5CartesianController(model, interface)
    controller.reset_to_home()

    # Keep current orientation, move EE to (0.243, 0.0, 0.317) over 2 s
    curr_pose = controller.get_eef_state().pose_6d().copy()
    curr_pose[:3] = np.array([0.243, 0.0, 0.317])
    robot_config = controller.get_robot_config()
    gripper_width = robot_config.gripper_width
    target = EEFState(curr_pose, gripper_width)
    target.timestamp = controller.get_timestamp() + 2.0
    controller.set_eef_cmd(target)

    target_pose_6d = controller.get_home_pose().copy()
    eef_reference_pose = None  # will be set when A is pressed
    preview_time = 0.05        # seconds ahead of current time
    pos_scale = 1.0            # scale controller meters -> arm meters (tune)  ### NEW
    gripper_min = 0.01
    gripper_command = gripper_width
    gripper_hold = False
    torque_limit = 0.55
    # -------------------------------------------------------------------

    right_engaged = False
    have_reference = False
    right_reference_pose = pysurvive.SurvivePose()
    right_reference_pose_inv = pysurvive.SurvivePose()
    _set_identity(right_reference_pose)
    _set_identity(right_reference_pose_inv)

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
                if not right_engaged:
                    continue
                pose_event = pysurvive.survive_simple_get_pose_updated_event(ctypes.byref(event))
                if not pose_event or not pose_event.contents.object:
                    continue
                obj = pose_event.contents.object
                subtype = pysurvive.survive_simple_object_get_subtype(obj)
                if subtype != pysurvive.SURVIVE_OBJECT_SUBTYPE_KNUCKLES_R:
                    continue

                pose = _pose_from_event(pose_event.contents.pose)
                if have_reference:
                    # pose is now "delta" in controller frame
                    pose = _apply_pose(right_reference_pose_inv, pose)

                    # --- NEW: use delta pose to command ARX5 -----------------
                    if eef_reference_pose is not None:
                        joint_state = controller.get_joint_state()
                        if not gripper_hold and abs(joint_state.gripper_torque) > torque_limit:
                            gripper_command = max(
                                gripper_min, min(gripper_width, joint_state.gripper_pos)
                            )
                            gripper_hold = True
                            print(
                                f"Gripper torque {joint_state.gripper_torque:.3f} Nm exceeding limit; holding at {gripper_command:.4f} m"
                            )

                        # simple position-only mapping L5-frame ~= world-frame
                        dx, dy, dz = pose.Pos[0], pose.Pos[1], pose.Pos[2]

                        target_pose_6d[:] = eef_reference_pose  # start from ref EE pose
                        # target_pose_6d[0] += pos_scale * dx
                        # target_pose_6d[1] += pos_scale * dy
                        target_pose_6d[0] += -pos_scale * dy
                        target_pose_6d[1] += -pos_scale * dx
                        target_pose_6d[2] += -pos_scale * dz
                        roll, pitch, yaw = _quat_to_euler(pose.Rot)
                        target_pose_6d[3] = eef_reference_pose[3] - pitch
                        target_pose_6d[4] = eef_reference_pose[4] - roll #intuitive pitch
                        target_pose_6d[5] = eef_reference_pose[5] - yaw

                        eef_cmd = EEFState()
                        eef_cmd.pose_6d()[:] = target_pose_6d
                        eef_cmd.gripper_pos = gripper_command
                        eef_cmd.timestamp = controller.get_timestamp() + preview_time
                        controller.set_eef_cmd(eef_cmd)
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
                if subtype != pysurvive.SURVIVE_OBJECT_SUBTYPE_KNUCKLES_R:
                    continue

                for i in range(button_event.contents.axis_count):
                    axis_id = button_event.contents.axis_ids[i]
                    if axis_id == pysurvive.SURVIVE_AXIS_TRIGGER:
                        trigger_val = button_event.contents.axis_val[i]
                        trigger_val = max(0.0, min(1.0, trigger_val))
                        gripper_command = gripper_width * (1.0 - trigger_val)
                        if gripper_command < gripper_min:
                            gripper_command = gripper_min
                        gripper_hold = False

                if button_event.contents.button_id == pysurvive.SURVIVE_BUTTON_A:
                    if button_event.contents.event_type == pysurvive.SURVIVE_INPUT_EVENT_BUTTON_DOWN:
                        right_engaged = True
                        pysurvive.survive_simple_object_get_latest_pose(
                            obj, ctypes.byref(right_reference_pose)
                        )
                        inv = _invert_pose(right_reference_pose)
                        _copy_pose(right_reference_pose_inv, inv)
                        have_reference = True

                        # snapshot current EE pose as reference                  ### NEW
                        eef_state = controller.get_eef_state()
                        eef_reference_pose = eef_state.pose_6d().copy()
                        print("Right A button engaged")

                    elif button_event.contents.event_type == pysurvive.SURVIVE_INPUT_EVENT_BUTTON_UP:
                        right_engaged = False
                        have_reference = False
                        eef_reference_pose = None     # forget ref on release   ### NEW
                        gripper_hold = False
                        _set_identity(right_reference_pose)
                        _set_identity(right_reference_pose_inv)
                        print("Right A button released")

            elif event_type == pysurvive.SurviveSimpleEventType_DeviceAdded:
                obj_event = pysurvive.survive_simple_get_object_event(ctypes.byref(event))
                if obj_event and obj_event.contents.object:
                    name = pysurvive.survive_simple_object_name(obj_event.contents.object).decode("utf-8")
                    print(f"({obj_event.contents.time:.3f}) Found '{name}'")
    finally:
        print("Cleaning up")
        pysurvive.survive_simple_close(ctx.ptr)
        try:
            controller.set_to_damping()
        except Exception as e:
            print(f"Error setting damping: {e}")


if __name__ == "__main__":
    main()
