#!/usr/bin/env python3
import math
import signal
import sys
import os
import threading
import time
import ctypes
from dataclasses import dataclass

import numpy as np
# pyopenxr installs its Python package under the name "xr"
import xr

# --- ARX5 SDK hookup -------------------------------------------------
SDK_ROOT = "/home/griffin/arx5-sdk"
sys.path.append(f"{SDK_ROOT}/python")
os.environ["LD_LIBRARY_PATH"] = f"{SDK_ROOT}/lib/x86_64:" + os.environ.get("LD_LIBRARY_PATH", "")

from arx5_interface import Arx5CartesianController, EEFState, RobotConfigFactory, ControllerConfigFactory
# ---------------------------------------------------------------------

# Arms are mounted upside down, so mirror Y/Z offsets relative to the captured pose.
UPSIDE_DOWN_SIGNS = (1.0, -1.0, -1.0)

# Action names
ENGAGE_ACTION_NAME = "a_click"
SHUTDOWN_ACTION_NAME = "b_click"


def _quat_to_euler(quat_wxyz):
    w, x, y, z = quat_wxyz
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


def _rotmat_to_quat_wxyz(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm > 1e-12:
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return (w, x, y, z)


def _pose_mat4_to_pos_quat(mat4):
    pos = mat4[:3, 3].copy()
    quat = _rotmat_to_quat_wxyz(mat4[:3, :3])
    return pos, quat


def _posef_to_mat4(posef):
    """xr.Posef -> 4x4 numpy matrix."""
    ox, oy, oz, ow = posef.orientation.x, posef.orientation.y, posef.orientation.z, posef.orientation.w
    px, py, pz = posef.position.x, posef.position.y, posef.position.z
    xx, yy, zz = ox * ox, oy * oy, oz * oz
    xy, xz, yz = ox * oy, ox * oz, oy * oz
    wx, wy, wz = ow * ox, ow * oy, ow * oz

    R = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy)],
        [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[0, 3] = px
    T[1, 3] = py
    T[2, 3] = pz
    return T


def _apply_upside_down_transform(target_pose, reference_pose, start_idx):
    if reference_pose is None:
        return
    for axis, sign in enumerate(UPSIDE_DOWN_SIGNS):
        if sign == 1.0:
            continue
        idx = start_idx + axis
        offset = target_pose[idx] - reference_pose[idx]
        target_pose[idx] = reference_pose[idx] + sign * offset


def _bool_edge_down(prev, curr):
    return (not prev) and curr


def _bool_edge_up(prev, curr):
    return prev and (not curr)


def _get_action_bool(session, action, subaction_path):
    state = xr.get_action_state_boolean(
        session,
        xr.ActionStateGetInfo(action=action, subaction_path=subaction_path),
    )
    if not state.is_active:
        return False
    return bool(state.current_state)


def _xr_time_now(instance):
    try:
        ts = xr.timespec()
        now_ns = time.time_ns()
        ts.tv_sec = now_ns // 1_000_000_000
        ts.tv_nsec = now_ns % 1_000_000_000
        result = xr.convert_timespec_time_to_time_khr(instance, ts)
        # Result may be bytes or int depending on pyopenxr version
        if isinstance(result, bytes):
            return int.from_bytes(result, byteorder='little', signed=False)
        return int(result)
    except Exception as e:
        # Debug the error on first call
        if not hasattr(_xr_time_now, '_logged_error'):
            print(f"[DEBUG] _xr_time_now failed: {e}")
            _xr_time_now._logged_error = True
        return 0


def _resolve_display_time(xr_ctx, frame_state):
    # Try to get the predicted display time directly as an integer
    try:
        if hasattr(frame_state.predicted_display_time, 'value'):
            display_time = int(frame_state.predicted_display_time.value)
        else:
            display_time = int(frame_state.predicted_display_time)
    except (ValueError, TypeError):
        # If it's bytes or some other type, try to convert
        pdt = frame_state.predicted_display_time
        if isinstance(pdt, bytes):
            display_time = int.from_bytes(pdt, byteorder='little', signed=False)
        else:
            display_time = 0

    if display_time > 0:
        return display_time

    # Fallback to timespec conversion if available
    if xr_ctx.has_timespec_ext:
        display_time = int(_xr_time_now(xr_ctx.instance))

    return display_time


@dataclass
class XRHandles:
    instance: xr.Instance
    system_id: int
    session: xr.Session
    space_local: xr.Space
    action_set: xr.ActionSet
    act_pose: xr.Action
    act_engage: xr.Action
    act_shutdown: xr.Action
    space_left: xr.Space
    space_right: xr.Space
    left_path: xr.Path
    right_path: xr.Path
    graphics_binding: object = None
    gl_context: object = None
    has_timespec_ext: bool = False
    act_trigger: xr.Action = None


class XRHandState:
    def __init__(self, name):
        self.name = name
        self.engaged = False
        self.have_reference = False
        self.ref_T = np.eye(4, dtype=np.float64)
        self.ref_T_inv = np.eye(4, dtype=np.float64)
        self.prev_a = False
        self.prev_b = False
        self.pose_valid = False


class ArmTeleopState:
    def __init__(self, name, controller):
        self.name = name
        self.controller = controller
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
        self.eef_reference_pose = None


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


def _create_openxr_session():
    app_info = xr.ApplicationInfo(application_name="teleop_openxr", engine_name="pyopenxr")

    try:
        ext_props = xr.enumerate_instance_extension_properties()
        ext_names = [
            p.extension_name.decode() if isinstance(p.extension_name, (bytes, bytearray)) else p.extension_name
            for p in ext_props
        ]
    except xr.exception.RuntimeUnavailableError as e:
        raise RuntimeError(
            "OpenXR runtime not found. Launch SteamVR and set it as the active OpenXR runtime "
            "(SteamVR Settings → Developer → Set SteamVR as OpenXR Runtime)."
        ) from e
    except Exception:
        ext_names = []

    enabled_exts = []
    if "XR_MND_headless" in ext_names:
        enabled_exts.append("XR_MND_headless")
    if xr.KHR_OPENGL_ENABLE_EXTENSION_NAME in ext_names:
        enabled_exts.append(xr.KHR_OPENGL_ENABLE_EXTENSION_NAME)
    if xr.KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME in ext_names:
        enabled_exts.append(xr.KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME)

    if ext_names:
        candidate_ext_lists = [enabled_exts]
    else:
        candidate_ext_lists = [
            ["XR_MND_headless", xr.KHR_OPENGL_ENABLE_EXTENSION_NAME, xr.KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME],
            ["XR_MND_headless", xr.KHR_OPENGL_ENABLE_EXTENSION_NAME],
            ["XR_MND_headless", xr.KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME],
            [xr.KHR_OPENGL_ENABLE_EXTENSION_NAME, xr.KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME],
            ["XR_MND_headless"],
            [xr.KHR_OPENGL_ENABLE_EXTENSION_NAME],
            [xr.KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME],
            [],
        ]

    instance = None
    last_ext_err = None
    for ext_list in candidate_ext_lists:
        try:
            create_info = xr.InstanceCreateInfo(
                application_info=app_info,
                enabled_extension_names=ext_list,
            )
            instance = xr.create_instance(create_info)
            enabled_exts = ext_list
            break
        except xr.exception.ExtensionNotPresentError as e:
            last_ext_err = e
    if instance is None:
        raise last_ext_err

    system_id = xr.get_system(instance, xr.SystemGetInfo(form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY))
    has_timespec_ext = xr.KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME in enabled_exts

    gl_context = None
    graphics_binding = None
    try:
        # Headless session may work on newer SteamVR builds; if this raises, a graphics binding is required.
        session = xr.create_session(instance, xr.SessionCreateInfo(system_id=system_id))
    except xr.exception.GraphicsDeviceInvalidError:
        if xr.KHR_OPENGL_ENABLE_EXTENSION_NAME not in enabled_exts:
            raise RuntimeError(
                "SteamVR rejected a headless session and XR_KHR_opengl_enable is unavailable. "
                "Update SteamVR or enable headless support, or provide a graphics binding."
            )
        from xr.utils.gl import create_graphics_binding
        from xr.utils.gl.glfw_util import GLFWOffscreenContextProvider

        gl_context = GLFWOffscreenContextProvider()
        graphics_binding = create_graphics_binding(gl_context)
        session = xr.create_session(
            instance,
            xr.SessionCreateInfo(system_id=system_id, next=graphics_binding.pointer),
        )

    space_local = xr.create_reference_space(
        session,
        xr.ReferenceSpaceCreateInfo(
            reference_space_type=xr.ReferenceSpaceType.LOCAL,
            pose_in_reference_space=xr.Posef(),
        ),
    )

    action_set = xr.create_action_set(
        instance,
        xr.ActionSetCreateInfo(
            action_set_name="teleop",
            localized_action_set_name="teleop",
            priority=0,
        ),
    )
    left_path = xr.string_to_path(instance, "/user/hand/left")
    right_path = xr.string_to_path(instance, "/user/hand/right")

    act_pose = xr.create_action(
        action_set,
        xr.ActionCreateInfo(
            action_name="hand_pose",
            action_type=xr.ActionType.POSE_INPUT,
            subaction_paths=[left_path, right_path],
            localized_action_name="hand pose",
        ),
    )
    act_engage = xr.create_action(
        action_set,
        xr.ActionCreateInfo(
            action_name=ENGAGE_ACTION_NAME,
            action_type=xr.ActionType.BOOLEAN_INPUT,
            subaction_paths=[left_path, right_path],
            localized_action_name="A click",
        ),
    )
    act_shutdown = xr.create_action(
        action_set,
        xr.ActionCreateInfo(
            action_name=SHUTDOWN_ACTION_NAME,
            action_type=xr.ActionType.BOOLEAN_INPUT,
            subaction_paths=[left_path, right_path],
            localized_action_name="B click",
        ),
    )
    act_trigger = xr.create_action(
        action_set,
        xr.ActionCreateInfo(
            action_name="trigger",
            action_type=xr.ActionType.FLOAT_INPUT,
            subaction_paths=[left_path, right_path],
            localized_action_name="Trigger",
        ),
    )

    space_left = xr.create_action_space(
        session,
        xr.ActionSpaceCreateInfo(
            action=act_pose,
            subaction_path=left_path,
            pose_in_action_space=xr.Posef(),
        ),
    )
    space_right = xr.create_action_space(
        session,
        xr.ActionSpaceCreateInfo(
            action=act_pose,
            subaction_path=right_path,
            pose_in_action_space=xr.Posef(),
        ),
    )

    # Suggest bindings for Valve Index controllers (Knuckles)
    valve_index_profile = xr.string_to_path(instance, "/interaction_profiles/valve/index_controller")
    index_bindings = [
        xr.ActionSuggestedBinding(act_pose, xr.string_to_path(instance, "/user/hand/left/input/grip/pose")),
        xr.ActionSuggestedBinding(act_pose, xr.string_to_path(instance, "/user/hand/right/input/grip/pose")),
        xr.ActionSuggestedBinding(act_engage, xr.string_to_path(instance, "/user/hand/left/input/a/click")),
        xr.ActionSuggestedBinding(act_engage, xr.string_to_path(instance, "/user/hand/right/input/a/click")),
        xr.ActionSuggestedBinding(act_shutdown, xr.string_to_path(instance, "/user/hand/left/input/b/click")),
        xr.ActionSuggestedBinding(act_shutdown, xr.string_to_path(instance, "/user/hand/right/input/b/click")),
        xr.ActionSuggestedBinding(act_trigger, xr.string_to_path(instance, "/user/hand/left/input/trigger/value")),
        xr.ActionSuggestedBinding(act_trigger, xr.string_to_path(instance, "/user/hand/right/input/trigger/value")),
    ]
    xr.suggest_interaction_profile_bindings(
        instance,
        xr.InteractionProfileSuggestedBinding(
            interaction_profile=valve_index_profile,
            suggested_bindings=index_bindings,
        ),
    )

    # Also suggest bindings for generic controller profile as fallback
    simple_profile = xr.string_to_path(instance, "/interaction_profiles/khr/simple_controller")
    simple_bindings = [
        xr.ActionSuggestedBinding(act_pose, xr.string_to_path(instance, "/user/hand/left/input/grip/pose")),
        xr.ActionSuggestedBinding(act_pose, xr.string_to_path(instance, "/user/hand/right/input/grip/pose")),
        xr.ActionSuggestedBinding(act_engage, xr.string_to_path(instance, "/user/hand/left/input/select/click")),
        xr.ActionSuggestedBinding(act_engage, xr.string_to_path(instance, "/user/hand/right/input/select/click")),
        xr.ActionSuggestedBinding(act_shutdown, xr.string_to_path(instance, "/user/hand/left/input/menu/click")),
        xr.ActionSuggestedBinding(act_shutdown, xr.string_to_path(instance, "/user/hand/right/input/menu/click")),
    ]
    xr.suggest_interaction_profile_bindings(
        instance,
        xr.InteractionProfileSuggestedBinding(
            interaction_profile=simple_profile,
            suggested_bindings=simple_bindings,
        ),
    )

    xr.attach_session_action_sets(
        session,
        xr.SessionActionSetsAttachInfo(action_sets=[action_set]),
    )

    return XRHandles(
        instance=instance,
        system_id=system_id,
        session=session,
        space_local=space_local,
        action_set=action_set,
        act_pose=act_pose,
        act_engage=act_engage,
        act_shutdown=act_shutdown,
        space_left=space_left,
        space_right=space_right,
        left_path=left_path,
        right_path=right_path,
        graphics_binding=graphics_binding,
        gl_context=gl_context,
        has_timespec_ext=has_timespec_ext,
        act_trigger=act_trigger,
    )


def run_teleop(ctx_args=None, status_callback=None, stop_event=None, install_signal_handlers=True):
    if stop_event is None:
        stop_event = threading.Event()

    # --- init OpenXR / SteamVR tracking -------------------------------
    xr_ctx = _create_openxr_session()
    print("OpenXR session started (SteamVR should be the active runtime).")
    # ------------------------------------------------------------------

    # --- init ARX5 controllers ---------------------------------------
    interfaces = {
        # Swap controllers: left knuckles drive the right arm (can1) and vice versa.
        "left": ("right", "can1"),
        "right": ("left", "can0"),
    }

    arm_states = {}
    controllers = {}
    for hand, (arm_name, interface) in interfaces.items():
        robot_cfg, ctrl_cfg = _create_controller(arm_name)
        controller = Arx5CartesianController(robot_cfg, ctrl_cfg, interface)
        _initialize_controller_state(controller)
        controllers[hand] = controller
        arm_states[hand] = ArmTeleopState(arm_name, controller)
        gain = controller.get_gain()
        print(
            f"{arm_name.capitalize()} initial gains:\n"
            f"  kp: {np.array2string(gain.kp(), precision=4)}\n"
            f"  kd: {np.array2string(gain.kd(), precision=4)}"
        )

    # Hand states
    xr_left = XRHandState("left")
    xr_right = XRHandState("right")

    session_state = xr.SessionState.IDLE
    interaction_profile_checked = False
    last_session_state = None
    session_begun = False

    preview_time = 0.05        # seconds ahead of current time
    pos_scale = 1.25           # scale controller meters -> arm meters (tune)
    workspace_limit = 0.7      # +/- 70 cm relative to captured pose
    slew_limit = 0.01          # limit per-update delta to 1 cm
    rot_scale = 1.25
    torque_limit = 0.80

    def request_stop():
        stop_event.set()

    if install_signal_handlers and threading.current_thread() is threading.main_thread():
        def stop_handler(signum, frame):
            if stop_event.is_set():
                raise SystemExit(1)
            stop_event.set()

        signal.signal(signal.SIGINT, stop_handler)
        signal.signal(signal.SIGTERM, stop_handler)

    def publish_status():
        if not status_callback:
            return
        for hand_state in arm_states.values():
            pose = hand_state.controller.get_eef_state().pose_6d().copy()
            status_callback(hand_state.name, pose)

    def poll_xr_events():
        nonlocal session_state, last_session_state, session_begun, interaction_profile_checked
        while True:
            try:
                event_buffer = xr.poll_event(xr_ctx.instance)
            except xr.EventUnavailable:
                break
            event_type = xr.StructureType(event_buffer.type)
            if event_type != xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED:
                continue
            event = ctypes.cast(
                ctypes.byref(event_buffer),
                ctypes.POINTER(xr.EventDataSessionStateChanged)
            ).contents
            session_state = xr.SessionState(event.state)
            if session_state != last_session_state:
                print(f"OpenXR session state: {session_state.name}")
                last_session_state = session_state
            if session_state == xr.SessionState.READY and not session_begun:
                xr.begin_session(
                    xr_ctx.session,
                    xr.SessionBeginInfo(xr.ViewConfigurationType.PRIMARY_STEREO)
                )
                session_begun = True
            elif session_state == xr.SessionState.STOPPING:
                try:
                    xr.end_session(xr_ctx.session)
                except Exception:
                    pass
                session_begun = False
            elif session_state in (xr.SessionState.EXITING, xr.SessionState.LOSS_PENDING):
                request_stop()

    try:
        publish_status()

        frame_counter = 0
        print("[LOOP] Entering main loop...")
        while not stop_event.is_set():
            poll_xr_events()
            if session_state not in (
                xr.SessionState.SYNCHRONIZED,
                xr.SessionState.VISIBLE,
                xr.SessionState.FOCUSED,
            ):
                time.sleep(0.01)
                continue

            frame_counter += 1
            # if frame_counter == 1:
            #     print(f"[LOOP] First frame in {session_state.name} state")
            # if frame_counter % 300 == 0:  # Every ~5 seconds
            #     print(f"[FRAME DEBUG] session_state={session_state.name}, frame={frame_counter}")

            # Check interaction profile once when focused (wait a bit for controllers to bind)
            if session_state == xr.SessionState.FOCUSED and not interaction_profile_checked:
                if not hasattr(run_teleop, 'focus_time'):
                    run_teleop.focus_time = time.time()

                # Wait 1 second after focus for controllers to bind
                if time.time() - run_teleop.focus_time > 1.0:
                    try:
                        print("\n=== Controller Detection Status ===")
                        for hand_name, hand_path in [("left", xr_ctx.left_path), ("right", xr_ctx.right_path)]:
                            profile_state = xr.get_current_interaction_profile(xr_ctx.session, hand_path)
                            if profile_state.interaction_profile != xr.NULL_PATH:
                                profile_path_str = xr.path_to_string(xr_ctx.instance, profile_state.interaction_profile)
                                print(f"{hand_name.capitalize()} hand using interaction profile: {profile_path_str}")
                            else:
                                print(f"{hand_name.capitalize()} hand: no interaction profile bound")
                                print(f"  → SteamVR may not have the controllers paired for input")
                                print(f"  → Try: SteamVR Settings → Controllers → Show Old Binding UI")
                        print("===================================\n")
                        interaction_profile_checked = True
                    except Exception as e:
                        print(f"Could not query interaction profile: {e}")
                        interaction_profile_checked = True

            try:
                frame_state = xr.wait_frame(xr_ctx.session)
                display_time = _resolve_display_time(xr_ctx, frame_state)
            except Exception as e:
                print(f"[ERROR] wait_frame or resolve_display_time failed: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.01)
                continue

            if display_time <= 0:
                # For headless sessions without valid display time, use current time in nanoseconds
                # OpenXR times are in nanoseconds, so use time.time_ns() for minimal latency
                display_time = time.time_ns()

            try:
                xr.begin_frame(xr_ctx.session)
            except xr.exception.FrameDiscarded:
                # Frame was discarded, we still need to end it
                try:
                    xr.end_frame(
                        xr_ctx.session,
                        xr.FrameEndInfo(
                            display_time=display_time,
                            environment_blend_mode=xr.EnvironmentBlendMode.OPAQUE,
                            layers=[],
                        ),
                    )
                except:
                    pass
                continue
            except Exception as e:
                print(f"[ERROR] begin_frame failed: {e}")
                continue
            xr.sync_actions(
                xr_ctx.session,
                xr.ActionsSyncInfo(
                    active_action_sets=[xr.ActiveActionSet(xr_ctx.action_set, xr.NULL_PATH)]
                ),
            )

            # Locate poses using the valid display_time we determined above
            try:
                locate_left = xr.locate_space(
                    xr_ctx.space_left, xr_ctx.space_local, display_time
                )
                locate_right = xr.locate_space(
                    xr_ctx.space_right, xr_ctx.space_local, display_time
                )
            except xr.exception.TimeInvalidError:
                # Time is still invalid, skip this frame
                if frame_counter < 10:
                    print(f"[WARNING] TimeInvalidError with display_time={display_time}, skipping frame")
                time.sleep(0.001)
                continue

            locs = {
                "left": locate_left,
                "right": locate_right,
            }
            hands = {
                "left": xr_left,
                "right": xr_right,
            }

            for hand_name, xr_hand in hands.items():
                loc = locs[hand_name]
                pose_valid = bool(
                    (loc.location_flags & xr.SpaceLocationFlags.POSITION_VALID_BIT)
                    and (loc.location_flags & xr.SpaceLocationFlags.ORIENTATION_VALID_BIT)
                )
                was_valid = xr_hand.pose_valid
                xr_hand.pose_valid = pose_valid
                if pose_valid and not was_valid:
                    print(f"{hand_name.capitalize()} controller tracking is active")
                elif was_valid and not pose_valid:
                    print(f"{hand_name.capitalize()} controller tracking lost; ensure visibility to SteamVR")

                subaction_path = xr_ctx.left_path if hand_name == "left" else xr_ctx.right_path
                a_state = _get_action_bool(xr_ctx.session, xr_ctx.act_engage, subaction_path)
                b_state = _get_action_bool(xr_ctx.session, xr_ctx.act_shutdown, subaction_path)

                # Read trigger
                arm = arm_states[hand_name]
                if xr_ctx.act_trigger:
                    trigger_state = xr.get_action_state_float(
                        xr_ctx.session,
                        xr.ActionStateGetInfo(action=xr_ctx.act_trigger, subaction_path=subaction_path),
                    )
                    if trigger_state.is_active:
                        trigger_val = max(0.0, min(1.0, trigger_state.current_state))
                        arm.gripper_command = arm.gripper_width * (1.0 - trigger_val)
                        if arm.gripper_command < arm.gripper_min:
                            arm.gripper_command = arm.gripper_min
                        arm.gripper_hold = False

                # Debug: log button state periodically (disabled)
                # if not hasattr(xr_hand, 'debug_counter'):
                #     xr_hand.debug_counter = 0
                # xr_hand.debug_counter += 1
                # if xr_hand.debug_counter % 300 == 0:  # Every ~5 seconds
                #     print(f"[DEBUG] {hand_name}: a={a_state}, b={b_state}, pose_valid={pose_valid}")

                if _bool_edge_down(xr_hand.prev_b, b_state):
                    print(f"{hand_name.capitalize()} shutdown button pressed — initiating shutdown")
                    request_stop()

                if pose_valid and _bool_edge_down(xr_hand.prev_a, a_state):
                    xr_hand.engaged = True
                    T = _posef_to_mat4(loc.pose)
                    xr_hand.ref_T = T
                    xr_hand.ref_T_inv = np.linalg.inv(T)
                    xr_hand.have_reference = True

                    arm = arm_states[hand_name]
                    eef_state = arm.controller.get_eef_state()
                    arm.eef_reference_pose = eef_state.pose_6d().copy()
                    arm.last_command_pose = arm.eef_reference_pose.copy()
                    arm.torque_limited = False
                    print(f"{hand_name.capitalize()} engage pressed")

                if _bool_edge_up(xr_hand.prev_a, a_state):
                    xr_hand.engaged = False
                    xr_hand.have_reference = False
                    xr_hand.ref_T = np.eye(4)
                    xr_hand.ref_T_inv = np.eye(4)
                    arm = arm_states[hand_name]
                    arm.eef_reference_pose = None
                    arm.gripper_hold = False
                    arm.last_command_pose = arm.controller.get_eef_state().pose_6d().copy()
                    arm.torque_limited = False
                    print(f"{hand_name.capitalize()} engage released")

                xr_hand.prev_a = bool(a_state)
                xr_hand.prev_b = bool(b_state)

                if not (xr_hand.engaged and xr_hand.have_reference and pose_valid):
                    continue

                controller = arm.controller
                if arm.eef_reference_pose is None:
                    continue

                T_cur = _posef_to_mat4(loc.pose)
                T_delta = xr_hand.ref_T_inv @ T_cur
                dpos, dquat = _pose_mat4_to_pos_quat(T_delta)

                joint_state = controller.get_joint_state()
                joint_torques = joint_state.torque().copy()
                abs_torque = np.abs(joint_torques)
                over_limit = abs_torque - arm.joint_torque_limit
                exceeding = np.where(over_limit > 0)[0]
                if exceeding.size > 0:
                    idxj = exceeding[np.argmax(over_limit[exceeding])]
                    actual_val = abs_torque[idxj]
                    limit_val = arm.joint_torque_limit[idxj]
                    if not arm.torque_limited:
                        print(
                            f"{arm.name.capitalize()} joint torque limit exceeded "
                            f"(joint {idxj}: {actual_val:.4f} Nm > {limit_val:.4f} Nm); holding pose"
                        )
                        arm.torque_limited = True
                    continue
                elif arm.torque_limited:
                    print(f"{arm.name.capitalize()} torque back within limits; resuming motion")
                    arm.torque_limited = False

                current_gripper_torque = abs(joint_state.gripper_torque)
                if current_gripper_torque > torque_limit:
                    arm.gripper_command = max(
                        arm.gripper_min,
                        min(arm.gripper_width, joint_state.gripper_pos),
                    )
                    arm.gripper_hold = True
                    print(
                        f"{arm.name.capitalize()} gripper torque {joint_state.gripper_torque:.3f} Nm exceeding limit; "
                        f"holding at {arm.gripper_command:.4f} m"
                    )

                dx, dy, dz = float(dpos[0]), float(dpos[1]), float(dpos[2])
                arm.target_pose_6d[:] = arm.eef_reference_pose
                arm.target_pose_6d[0] += -pos_scale * dy
                arm.target_pose_6d[1] += -pos_scale * dx
                arm.target_pose_6d[2] += -pos_scale * dz
                _apply_upside_down_transform(arm.target_pose_6d, arm.eef_reference_pose, 0)

                workspace_clamped = False
                workspace_reason = ""
                for axis in range(3):
                    min_bound = arm.eef_reference_pose[axis] - workspace_limit
                    max_bound = arm.eef_reference_pose[axis] + workspace_limit
                    clamped_val = max(min_bound, min(max_bound, arm.target_pose_6d[axis]))
                    if clamped_val != arm.target_pose_6d[axis]:
                        if not workspace_clamped:
                            requested_offset = arm.target_pose_6d[axis] - arm.eef_reference_pose[axis]
                            workspace_reason = f"axis {axis}: {requested_offset:+.4f} m exceeds ±{workspace_limit:.4f} m"
                        workspace_clamped = True
                        arm.target_pose_6d[axis] = clamped_val
                if workspace_clamped and not arm.workspace_clamped:
                    print(f"{arm.name.capitalize()} workspace clamp active ({workspace_reason})")
                elif not workspace_clamped and arm.workspace_clamped:
                    print(f"{arm.name.capitalize()} workspace clamp cleared")
                arm.workspace_clamped = workspace_clamped

                if arm.last_command_pose is not None:
                    velocity_limited = False
                    velocity_reason = ""
                    for axis in range(3):
                        prev = arm.last_command_pose[axis]
                        delta = arm.target_pose_6d[axis] - prev
                        if abs(delta) > slew_limit:
                            arm.target_pose_6d[axis] = prev + math.copysign(slew_limit, delta)
                            velocity_limited = True
                            if not velocity_reason:
                                velocity_reason = f"axis {axis}: step {abs(delta):.4f} m > {slew_limit:.4f} m limit"
                    if velocity_limited and not arm.velocity_limited:
                        print(f"{arm.name.capitalize()} translation slew limit active ({velocity_reason})")
                    elif not velocity_limited and arm.velocity_limited:
                        print(f"{arm.name.capitalize()} translation slew limit cleared")
                    arm.velocity_limited = velocity_limited

                roll, pitch, yaw = _quat_to_euler(dquat)
                # Pitch and yaw flipped (positive), roll keeps original (negative)
                arm.target_pose_6d[3] = arm.eef_reference_pose[3] - rot_scale * pitch ## what i see as roll
                arm.target_pose_6d[4] = arm.eef_reference_pose[4] + rot_scale * roll
                arm.target_pose_6d[5] = arm.eef_reference_pose[5] + rot_scale * yaw

                eef_cmd = EEFState()
                eef_cmd.pose_6d()[:] = arm.target_pose_6d
                eef_cmd.gripper_pos = arm.gripper_command
                eef_cmd.timestamp = controller.get_timestamp() + preview_time
                controller.set_eef_cmd(eef_cmd)
                arm.last_command_pose = arm.target_pose_6d.copy()

            xr.end_frame(
                xr_ctx.session,
                xr.FrameEndInfo(
                    display_time=display_time,
                    environment_blend_mode=xr.EnvironmentBlendMode.OPAQUE,
                    layers=[],
                ),
            )
            publish_status()
            # Minimal sleep to prevent busy-waiting but maximize responsiveness
            time.sleep(0.0001)

    finally:
        print("Cleaning up")
        try:
            xr.end_session(xr_ctx.session)
            xr.destroy_space(xr_ctx.space_left)
            xr.destroy_space(xr_ctx.space_right)
            xr.destroy_space(xr_ctx.space_local)
            xr.destroy_session(xr_ctx.session)
            xr.destroy_instance(xr_ctx.instance)
        except Exception:
            pass
        try:
            if xr_ctx.gl_context is not None:
                xr_ctx.gl_context.destroy()
        except Exception:
            pass

        for hand_name, controller in controllers.items():
            label = arm_states[hand_name].name.capitalize()
            try:
                print(f"{label}: returning to home pose")
                controller.reset_to_home()
            except Exception as e:
                print(f"Error during shutdown for {label}: {e}")


def main():
    run_teleop()


if __name__ == "__main__":
    main()
