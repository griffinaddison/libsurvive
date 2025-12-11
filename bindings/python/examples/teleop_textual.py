#!/usr/bin/env python3
import importlib.util
import io
import os
import sys
import threading
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from textual import events
from textual.app import App, ComposeResult
from textual.containers import Grid
from textual.widgets import Button, Static


_TELEOP_PATH = Path(__file__).with_name("teleop-example.py")
_SPEC = importlib.util.spec_from_file_location("teleop_example_module", _TELEOP_PATH)
_TELEOP = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_TELEOP)


def _run_teleop(**kwargs):
    return _TELEOP.run_teleop(**kwargs)


class PoseDisplay(Static):
    def __init__(self, label: str, **kwargs):
        super().__init__(**kwargs)
        self.label = label
        self.set_pose([0.0] * 6)

    def set_pose(self, pose_6d):
        pose = (
            f"x: {pose_6d[0]: .3f}\n"
            f"y: {pose_6d[1]: .3f}\n"
            f"z: {pose_6d[2]: .3f}\n"
            f"roll:  {pose_6d[3]: .3f}\n"
            f"pitch: {pose_6d[4]: .3f}\n"
            f"yaw:   {pose_6d[5]: .3f}"
        )
        self.update(f"[b]{self.label}[/b]\n{pose}")


class TeleopTUI(App):
    CSS = """
    Screen {
        align: center middle;
        background: #111111;
    }
    #status-grid {
        grid-size: 2;
        grid-gutter: 2;
    }
    PoseDisplay {
        padding: 1 3;
        border: round $primary;
        height: 10;
        width: 40;
        background: #1f1f1f;
        color: #f0f0f0;
        content-align: left top;
        overflow: hidden;
    }
    #quit {
        margin-top: 1;
        width: 20;
    }
    """

    BINDINGS = [("q", "quit_app", "Quit")]

    def __init__(self):
        super().__init__()
        self.stop_event = threading.Event()
        self.worker = None

    def compose(self) -> ComposeResult:
        with Grid(id="status-grid"):
            yield PoseDisplay("Left", id="left_pose")
            yield PoseDisplay("Right", id="right_pose")
        yield Button("Quit", id="quit", variant="success")

    def on_mount(self) -> None:
        self.worker = self.run_worker(self._worker_entry, thread=True, exclusive=True)

    def _worker_entry(self) -> None:
        ctx_args = [sys.argv[0]]
        os.environ.setdefault("SURVIVE_LOG_LEVEL", "0")
        log_buffer = io.StringIO()
        try:
            with redirect_stdout(log_buffer), redirect_stderr(log_buffer):
                _run_teleop(
                    ctx_args=ctx_args,
                    status_callback=self._report_pose,
                    stop_event=self.stop_event,
                    install_signal_handlers=False,
                )
        except Exception as exc:  # pragma: no cover
            self.call_from_thread(self.bell)
            self.log.exception("teleop worker failed", exc_info=exc)
        finally:
            self.call_from_thread(self.action_quit_app)

    def _report_pose(self, hand_name: str, pose_6d):
        widget_id = "left_pose" if hand_name == "left" else "right_pose"
        self.call_from_thread(self._update_pose_widget, widget_id, pose_6d)

    def _update_pose_widget(self, widget_id: str, pose_6d) -> None:
        widget = self.query_one(f"#{widget_id}", PoseDisplay)
        widget.set_pose(pose_6d)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit":
            self.action_quit_app()

    def action_quit_app(self) -> None:
        self.stop_event.set()
        if self.worker:
            self.worker.cancel()
        self.exit()

    def on_unmount(self, event: events.Unmount) -> None:
        self.stop_event.set()


if __name__ == "__main__":
    TeleopTUI().run()
