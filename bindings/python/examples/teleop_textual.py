#!/usr/bin/env python3
import importlib.util
import sys
import threading
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from textual import events
from textual.app import App, ComposeResult
from textual.containers import Grid, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Log, Static


_TELEOP_PATH = Path(__file__).with_name("teleop-example.py")
_SPEC = importlib.util.spec_from_file_location("teleop_example_module", _TELEOP_PATH)
_TELEOP = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_TELEOP)


def _run_teleop(**kwargs):
    return _TELEOP.run_teleop(**kwargs)


class PoseDisplay(Static):
    pose = reactive("x: 0.000  y: 0.000  z: 0.000")

    def __init__(self, label: str, **kwargs):
        super().__init__(**kwargs)
        self.label = label

    def watch_pose(self, pose: str) -> None:
        self.update(f"[b]{self.label}[/b]\n{pose}")

    def set_pose(self, pose_6d):
        self.pose = f"x: {pose_6d[0]: .3f}  y: {pose_6d[1]: .3f}  z: {pose_6d[2]: .3f}"


class UILogWriter:
    def __init__(self, callback):
        self._callback = callback
        self._buffer = ""

    def write(self, data: str):
        if not data:
            return 0
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            text = line.rstrip()
            if text:
                self._callback(text)
        return len(data)

    def flush(self):
        if self._buffer.strip():
            self._callback(self._buffer.strip())
        self._buffer = ""


class TeleopTUI(App):
    CSS = """
    Screen {
        align: center middle;
    }
    #status-grid {
        grid-size: 2;
        grid-gutter: 2;
    }
    PoseDisplay {
        padding: 1 2;
        border: round $primary;
        height: 5;
        width: 30;
    }
    #teleop_log {
        height: 15;
        width: 70;
        border: round $secondary;
        margin-top: 1;
    }
    """

    BINDINGS = [("q", "quit_app", "Quit")]

    def __init__(self):
        super().__init__()
        self.stop_event = threading.Event()
        self.worker = None

    def compose(self) -> ComposeResult:
        with Vertical():
            with Grid(id="status-grid"):
                yield PoseDisplay("Left", id="left_pose")
                yield PoseDisplay("Right", id="right_pose")
            yield Log(id="teleop_log", highlight=False)
            yield Button("Quit", id="quit", variant="error")

    def on_mount(self) -> None:
        self.worker = self.run_worker(self._worker_entry, thread=True, exclusive=True)

    def _worker_entry(self) -> None:
        ctx_args = [sys.argv[0]]
        log_writer = UILogWriter(lambda msg: self.call_from_thread(self._append_log, msg))
        try:
            with redirect_stdout(log_writer), redirect_stderr(log_writer):
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
            log_writer.flush()
            self.call_from_thread(self.action_quit_app)

    def _report_pose(self, hand_name: str, pose_6d):
        widget_id = "left_pose" if hand_name == "left" else "right_pose"
        self.call_from_thread(self._update_pose_widget, widget_id, pose_6d)

    def _update_pose_widget(self, widget_id: str, pose_6d) -> None:
        widget = self.query_one(f"#{widget_id}", PoseDisplay)
        widget.set_pose(pose_6d)

    def _append_log(self, message: str) -> None:
        log = self.query_one("#teleop_log", Log)
        log.write(message)

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
