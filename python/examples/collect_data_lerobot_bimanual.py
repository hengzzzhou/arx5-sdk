"""
Bimanual data collection with RealSense cameras and tactile sensor in LeRobot dataset format.

Both arms enter teach mode simultaneously. Drag them by hand to demonstrate tasks.
Camera images and tactile sensor images are recorded alongside joint states.

Usage:
    # Full setup: 2 RealSense cameras + 4 tactile sensors
    python collect_data_lerobot_bimanual.py X5 can0 X5 can1 \
        --repo-id user/arx5_bimanual --task "pick and place" --camera \
        --tactile /dev/video12 --tactile /dev/video14 \
        --tactile /dev/video16 --tactile /dev/video18

    # With RealSense cameras only
    python collect_data_lerobot_bimanual.py X5 can0 X5 can1 \
        --repo-id user/arx5_bimanual --task "pick and place" --camera

    # Without any camera
    python collect_data_lerobot_bimanual.py X5 can0 X5 can1 \
        --repo-id user/arx5_bimanual --task "pick and place"

Controls (press in the running terminal):
    Enter - Toggle: start new episode / stop and save current episode
    q     - Quit and finalize dataset
"""

import os
import sys
import time
import select
import tty
import termios

import numpy as np
import click
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import arx5_interface as arx5

from lerobot.datasets.lerobot_dataset import LeRobotDataset

JOINT_NAMES_LEFT = [f"left_joint_{i}" for i in range(6)] + ["left_gripper"]
JOINT_NAMES_RIGHT = [f"right_joint_{i}" for i in range(6)] + ["right_gripper"]
JOINT_NAMES_ALL = JOINT_NAMES_LEFT + JOINT_NAMES_RIGHT  # 14 dims


class TerminalInput:
    """Non-blocking terminal input reader. Only captures keys typed in THIS terminal."""

    def __init__(self):
        self._fd = sys.stdin.fileno()
        self._old_settings = None

    def __enter__(self):
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *args):
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def get_key(self) -> str | None:
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


class RealSenseCamera:
    """RealSense camera wrapper for RGB capture."""

    def __init__(self, serial_number: str, width=640, height=480, fps=30):
        import pyrealsense2 as rs
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(serial_number)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        self.width = width
        self.height = height
        self.serial = serial_number

    def start(self):
        self.pipeline.start(self.config)
        # Wait a few frames for auto-exposure to settle
        for _ in range(30):
            self.pipeline.wait_for_frames()

    def stop(self):
        self.pipeline.stop()

    def get_frame(self) -> np.ndarray:
        """Get current RGB frame as (H, W, 3) uint8 numpy array."""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        return np.asanyarray(color_frame.get_data())


class USBCamera:
    """USB camera wrapper (for tactile sensors etc.) using OpenCV."""

    def __init__(self, device: str | int, width=640, height=480):
        import cv2
        self.device = device
        self.cap = cv2.VideoCapture(int(device) if str(device).isdigit() else device)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open USB camera: {device}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frame(self) -> np.ndarray:
        """Get current RGB frame as (H, W, 3) uint8 numpy array."""
        import cv2
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read from USB camera: {self.device}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def stop(self):
        self.cap.release()


def discover_realsense_cameras() -> list[str]:
    """Return serial numbers of all connected RealSense devices."""
    import pyrealsense2 as rs
    ctx = rs.context()
    return [dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()]


def create_dataset(repo_id: str, fps: int,
                   image_sources: dict[str, tuple[int, int]],
                   root: str | None = None) -> LeRobotDataset:
    """Create dataset with arbitrary image sources.

    Args:
        image_sources: dict mapping feature name -> (height, width)
    """
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": JOINT_NAMES_ALL,
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": JOINT_NAMES_ALL,
        },
    }
    for name, (h, w) in image_sources.items():
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root,
        robot_type="arx5_bimanual",
        use_videos=True,
    )


def get_state_vector(joint_state: arx5.JointState) -> np.ndarray:
    """Extract 7-dim state: 6 joint positions + gripper position."""
    pos = joint_state.pos().copy()
    gripper = np.array([joint_state.gripper_pos], dtype=np.float64)
    return np.concatenate([pos, gripper]).astype(np.float32)


def get_bimanual_state(left: arx5.Arx5JointController, right: arx5.Arx5JointController) -> np.ndarray:
    """Get concatenated 14-dim state from both arms: [left(7), right(7)]."""
    left_state = get_state_vector(left.get_joint_state())
    right_state = get_state_vector(right.get_joint_state())
    return np.concatenate([left_state, right_state])


def enter_teach_mode(controller: arx5.Arx5JointController, transition_time: float = 0.5):
    """Gradually transition to compliant teach mode to avoid jerking."""
    controller_config = controller.get_controller_config()
    dt = controller_config.controller_dt

    start_gain = controller.get_gain()
    start_kp = start_gain.kp().copy()
    start_kd = start_gain.kd().copy()
    start_gripper_kp = start_gain.gripper_kp
    start_gripper_kd = start_gain.gripper_kd

    target_kd = controller_config.default_kd * 0.3

    steps = max(int(transition_time / dt), 1)
    for i in range(1, steps + 1):
        alpha = i / steps
        gain = controller.get_gain()
        gain.kp()[:] = start_kp * (1.0 - alpha)
        gain.kd()[:] = start_kd * (1.0 - alpha) + target_kd * alpha
        gain.gripper_kp = start_gripper_kp * (1.0 - alpha)
        gain.gripper_kd = start_gripper_kd * (1.0 - alpha)  # gripper kd → 0, no resistance
        controller.set_gain(gain)
        time.sleep(dt)


def init_controller(model: str, interface: str) -> arx5.Arx5JointController:
    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", robot_config.joint_dof
    )
    controller_config.background_send_recv = True
    return arx5.Arx5JointController(robot_config, controller_config, interface)


@click.command()
@click.argument("left_model")       # Left arm model: X5 or L5
@click.argument("left_interface")   # Left arm CAN bus (can0 etc.)
@click.argument("right_model")      # Right arm model: X5 or L5
@click.argument("right_interface")  # Right arm CAN bus (can1 etc.)
@click.option("--repo-id", required=True, help="Dataset repo ID, e.g. user/arx5_bimanual")
@click.option("--task", default="pick and place", help="Task description for episodes")
@click.option("--fps", default=6, help="Recording frequency in Hz")
@click.option("--root", default=None, help="Dataset root directory")
@click.option("--camera", is_flag=True, default=False, help="Enable RealSense camera recording")
@click.option("--cam-width", default=640, help="Camera resolution width")
@click.option("--cam-height", default=480, help="Camera resolution height")
@click.option("--tactile", multiple=True, help="Tactile sensor device path(s), e.g. --tactile /dev/video12 --tactile /dev/video14")
def main(left_model: str, left_interface: str, right_model: str, right_interface: str,
         repo_id: str, task: str, fps: int, root: str | None,
         camera: bool, cam_width: int, cam_height: int, tactile: tuple[str, ...]):
    assert left_interface != right_interface, "Left and right arms must be on different CAN interfaces"

    np.set_printoptions(precision=4, suppress=True)

    # All image sources: name -> (object with get_frame()/stop(), height, width)
    image_sources: dict[str, object] = {}  # cam_name -> camera object

    # Initialize RealSense cameras if enabled
    if camera:
        serials = discover_realsense_cameras()
        if len(serials) == 0:
            print("WARNING: --camera enabled but no RealSense devices found.")
        else:
            for i, sn in enumerate(sorted(serials)):
                cam_name = f"cam_{i}" if len(serials) > 1 else "cam"
                print(f"Initializing {cam_name} (SN: {sn}, {cam_width}x{cam_height})...")
                c = RealSenseCamera(sn, width=cam_width, height=cam_height, fps=fps)
                c.start()
                image_sources[cam_name] = c
            print(f"{len(image_sources)} RealSense camera(s) ready.")

    # Initialize tactile sensors
    for i, tac_dev in enumerate(sorted(tactile)):
        tac_name = f"tactile_{i}" if len(tactile) > 1 else "tactile"
        print(f"Initializing {tac_name} ({tac_dev})...")
        tac = USBCamera(tac_dev)
        print(f"  {tac_name} ready: {tac.width}x{tac.height}")
        image_sources[tac_name] = tac

    # Initialize both arms
    print(f"Initializing left arm ({left_model} on {left_interface})...")
    left = init_controller(left_model, left_interface)
    print(f"Initializing right arm ({right_model} on {right_interface})...")
    right = init_controller(right_model, right_interface)

    left.reset_to_home()
    right.reset_to_home()

    dt = 1.0 / fps

    # Create dataset - collect image source dimensions
    source_dims = {name: (src.height, src.width) for name, src in image_sources.items()}
    dataset = create_dataset(repo_id, fps, source_dims, root)
    episode_count = 0

    print(f"\nDataset: {repo_id}")
    print(f"Task: '{task}'")
    print(f"Recording at {fps} Hz")
    print(f"State format: [left(7), right(7)] = 14 dims")
    if image_sources:
        for name, src in image_sources.items():
            print(f"  {name}: {src.width}x{src.height} RGB")
    print(f"\nControls (press in THIS terminal):")
    print(f"  Enter  - Start episode / Stop and save episode")
    print(f"  q      - Quit and finalize dataset\n")
    print("Ready. Press Enter to start first episode.")

    recording = False
    frame_count = 0

    try:
        with TerminalInput() as term:
            while True:
                key = term.get_key()
                if key is not None:
                    if key == "\n" or key == "\r":
                        if not recording:
                            # Enter teach mode on both arms
                            print("Entering teach mode...")
                            enter_teach_mode(left)
                            enter_teach_mode(right)
                            recording = True
                            frame_count = 0
                            episode_start_time = time.monotonic()
                            next_frame_time = episode_start_time
                            print(f"\n--- Episode {episode_count} started ---")
                            print("Both arms in teach mode. Move them freely.")
                            print("Press Enter to stop and save.")
                        else:
                            recording = False
                            dataset.save_episode()
                            print(f"\nEpisode {episode_count} saved ({frame_count} frames, {frame_count / fps:.1f}s)")
                            episode_count += 1
                            left.reset_to_home()
                            right.reset_to_home()
                            print(f"\nReady. Press Enter for next episode, 'q' to quit.")

                    elif key == "q":
                        if recording:
                            print("\nStopping current episode first...")
                            recording = False
                            dataset.save_episode()
                            print(f"Episode {episode_count} saved ({frame_count} frames)")
                            episode_count += 1

                        print(f"\nFinalizing dataset ({episode_count} episodes)...")
                        dataset.finalize()
                        left.reset_to_home()
                        right.reset_to_home()
                        print(f"Dataset saved to: {dataset.root}")
                        print("Done!")
                        return

                if recording:
                    # Record actual wall-clock timestamp
                    timestamp = time.monotonic() - episode_start_time

                    state = get_bimanual_state(left, right)
                    frame_data = {
                        "observation.state": torch.from_numpy(state),
                        "action": torch.from_numpy(state),
                        "task": task,
                    }
                    for src_name, src_obj in image_sources.items():
                        frame_data[f"observation.images.{src_name}"] = src_obj.get_frame()
                    # Inject real timestamp into episode_buffer after add_frame
                    dataset.add_frame(frame_data)
                    dataset.episode_buffer["timestamp"][-1] = timestamp
                    frame_count += 1
                    if frame_count % fps == 0:
                        print(f"  Recording... {timestamp:.0f}s ({frame_count} frames)", end="\r")
                    # Clock-aligned sleep: account for processing time
                    next_frame_time += dt
                    sleep_time = next_frame_time - time.monotonic()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    else:
                        next_frame_time = time.monotonic()
                else:
                    time.sleep(0.05)
    finally:
        for src_obj in image_sources.values():
            src_obj.stop()


if __name__ == "__main__":
    main()
