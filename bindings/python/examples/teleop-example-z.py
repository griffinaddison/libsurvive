#!/usr/bin/env python3
import ctypes
import signal
import sys
import os
import numpy as np

import pysurvive

# --- ARX5 SDK hookup -------------------------------------------------
SDK_ROOT = "/home/griffin/arx5-sdk"
sys.path.append(f"{SDK_ROOT}/python")
os.environ["LD_LIBRARY_PATH"] = f"{SDK_ROOT}/lib/x86_64:" + os.environ.get("LD_LIBRARY_PATH", "")

import arx5_interface as arx5
from arx5_interface import Arx5CartesianController, EEFState
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
        w1 * w2 - x1 * x2 - y1 * y1 - z1 * z2,
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


def main():
    # --- init libsurvive ----------------------------------------------
    ctx = pysurvive.SimpleContext(sys.argv)
    if not ctx.ptr:
        print("Failed to initialize libsurvive context", file=sys.stderr)
        sys.exit(1)

    for obj in ctx.Objects():
        print(obj.Name().decode("utf-8"))

    # --- init ARX5 (L5 on can0) ---------------------------------------
    model = "L5"
    interface = "can1"
    controller = Arx5CartesianController(model, interface)
    controller.reset_to_home()
    robot_config = controller.get_robot_config()
    gripper_width = robot_config.gripper_width

    # Keep current orientation, move EE to (0.243, 0.0, 0.317) over 2 s
    curr_pose = controller.get_eef_state().pose_6d().copy()
    curr_pose[:3] = np.array([0.243, 0.0, 0.317])
    target = EEFState(curr_pose, gripper_width / 2)
    target.timestamp = controller.get_timestamp() + 2.0
    controller.set_eef_cmd(target)



 # • set_eef_cmd will automatically fill in a timestamp if you leave it at zero, using controller_config.default_preview_time, so sending EEFState(pose,
 #    gripper, timestamp=time.time()+2.0) will make the arm take ~2 s to reach that pose.
 #    • set_eef_traj expects timestamps you choose; by spacing them farther apart you get slower, smoother motion across the full sequence. The controller
 #    seeds the trajectory with the current j
 #

    # lets move to 0.253, 0,  0.317 in 2 seconds




    # keep EE pose fixed at reference, only change gripper_pos
    eef_reference_pose = controller.get_home_pose().copy()
    gripper_reference = 0.0
    preview_time = 0.05  # s in the future
    gripper_scale = -2.0  # meters of controller z → fraction of full width
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
            event_type = pysurvive.survive_simple_wait_for_event(
                ctx.ptr, ctypes.byref(event)
            )
            if event_type == pysurvive.SurviveSimpleEventType_Shutdown:
                break
            if event_type == pysurvive.SurviveSimpleEventType_None:
                continue

            if event_type == pysurvive.SurviveSimpleEventType_PoseUpdateEvent:
                if not right_engaged:
                    continue
                pose_event = pysurvive.survive_simple_get_pose_updated_event(
                    ctypes.byref(event)
                )
                if not pose_event or not pose_event.contents.object:
                    continue
                obj = pose_event.contents.object
                subtype = pysurvive.survive_simple_object_get_subtype(obj)
                if subtype != pysurvive.SURVIVE_OBJECT_SUBTYPE_KNUCKLES_R:
                    continue

                pose = _pose_from_event(pose_event.contents.pose)
                if have_reference:
                    # Δpose in controller frame
                    pose = _apply_pose(right_reference_pose_inv, pose)

                    # use only Δz to drive gripper
                    dz = pose.Pos[2]  # meters in controller frame
                    # positive dz → open, negative → close
                    target_gripper = gripper_reference + gripper_scale * dz * gripper_width

                    # clamp
                    if target_gripper < 0.0:
                        target_gripper = 0.0
                    elif target_gripper > gripper_width:
                        target_gripper = gripper_width

                    # build command: fixed pose, varying gripper
                    eef_cmd = EEFState()
                    eef_cmd.pose_6d()[:] = eef_reference_pose
                    eef_cmd.gripper_pos = target_gripper
                    eef_cmd.timestamp = controller.get_timestamp() + preview_time
                    # print
                    print(
                        f"Setting gripper to {target_gripper:.4f} m (dz={dz:.4f} m)"
                    )

                    controller.set_eef_cmd(eef_cmd)

            elif event_type == pysurvive.SurviveSimpleEventType_ButtonEvent:
                button_event = pysurvive.survive_simple_get_button_event(
                    ctypes.byref(event)
                )
                if not button_event or not button_event.contents.object:
                    continue
                obj = button_event.contents.object
                subtype = pysurvive.survive_simple_object_get_subtype(obj)
                if subtype != pysurvive.SURVIVE_OBJECT_SUBTYPE_KNUCKLES_R:
                    continue

                if button_event.contents.button_id == pysurvive.SURVIVE_BUTTON_A:
                    if (
                        button_event.contents.event_type
                        == pysurvive.SURVIVE_INPUT_EVENT_BUTTON_DOWN
                    ):
                        right_engaged = True
                        pysurvive.survive_simple_object_get_latest_pose(
                            obj, ctypes.byref(right_reference_pose)
                        )
                        inv = _invert_pose(right_reference_pose)
                        _copy_pose(right_reference_pose_inv, inv)
                        have_reference = True

                        # snapshot EE pose + current gripper pos
                        eef_state = controller.get_eef_state()
                        eef_reference_pose = eef_state.pose_6d().copy()
                        gripper_reference = eef_state.gripper_pos

                        print("Right A button engaged (z controls gripper)")

                    elif (
                        button_event.contents.event_type
                        == pysurvive.SURVIVE_INPUT_EVENT_BUTTON_UP
                    ):
                        right_engaged = False
                        have_reference = False
                        _set_identity(right_reference_pose)
                        _set_identity(right_reference_pose_inv)
                        print("Right A button released")

            elif event_type == pysurvive.SurviveSimpleEventType_DeviceAdded:
                obj_event = pysurvive.survive_simple_get_object_event(
                    ctypes.byref(event)
                )
                if obj_event and obj_event.contents.object:
                    name = (
                        pysurvive.survive_simple_object_name(
                            obj_event.contents.object
                        ).decode("utf-8")
                    )
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

