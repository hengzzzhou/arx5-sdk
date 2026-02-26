"""
Collect teleoperation demonstration data in LeRobot dataset format.

Usage:
    python collect_data_lerobot.py X5 can0 --repo-id user/arx5_pick_place --task "pick and place"

Controls (press in the running terminal):
    Enter - Toggle: start new episode / stop and save current episode
    q + Enter - Quit and finalize dataset
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

JOINT_NAMES = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]


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
        """Return a key character if available, else None (non-blocking)."""
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


def create_dataset(repo_id: str, fps: int, root: str | None = None) -> LeRobotDataset:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": JOINT_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": JOINT_NAMES,
        },
    }
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root,
        robot_type="arx5",
        use_videos=False,
    )


def get_state_vector(joint_state: arx5.JointState) -> np.ndarray:
    """Extract 7-dim state: 6 joint positions + gripper position."""
    pos = joint_state.pos().copy()  # shape (6,)
    gripper = np.array([joint_state.gripper_pos], dtype=np.float64)
    return np.concatenate([pos, gripper]).astype(np.float32)


def enter_teach_mode(controller: arx5.Arx5JointController, transition_time: float = 0.5):
    """Gradually transition to compliant teach mode to avoid jerking.

    Instead of instantly dropping kp to 0, we ramp down kp and kd over
    transition_time seconds so gravity compensation can keep up.
    """
    controller_config = controller.get_controller_config()
    dt = controller_config.controller_dt

    # Get current gain (should have kp > 0 from reset_to_home)
    start_gain = controller.get_gain()
    start_kp = start_gain.kp().copy()
    start_kd = start_gain.kd().copy()
    start_gripper_kp = start_gain.gripper_kp
    start_gripper_kd = start_gain.gripper_kd

    # Target: kp=0, kd=default*0.3 (enough to damp but still easy to drag)
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


@click.command()
@click.argument("model")  # ARX arm model: X5 or L5
@click.argument("interface")  # CAN bus name (can0 etc.)
@click.option("--repo-id", required=True, help="Dataset repo ID, e.g. user/arx5_pick_place")
@click.option("--task", default="pick and place", help="Task description for episodes")
@click.option("--fps", default=50, help="Recording frequency in Hz")
@click.option("--root", default=None, help="Dataset root directory (default: lerobot default)")
def main(model: str, interface: str, repo_id: str, task: str, fps: int, root: str | None):
    # Initialize robot
    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", robot_config.joint_dof
    )
    controller_config.background_send_recv = True
    controller = arx5.Arx5JointController(robot_config, controller_config, interface)
    np.set_printoptions(precision=4, suppress=True)

    controller.reset_to_home()
    dt = 1.0 / fps

    # Create dataset
    dataset = create_dataset(repo_id, fps, root)
    episode_count = 0

    print(f"\nDataset: {repo_id}")
    print(f"Task: '{task}'")
    print(f"Recording at {fps} Hz\n")
    print("Controls (press in THIS terminal):")
    print("  Enter  - Start episode / Stop and save episode")
    print("  q      - Quit and finalize dataset\n")
    print("Ready. Press Enter to start first episode.")

    recording = False
    frame_count = 0

    with TerminalInput() as term:
        while True:
            key = term.get_key()
            if key is not None:
                if key == "\n" or key == "\r":  # Enter key
                    if not recording:
                        enter_teach_mode(controller)
                        recording = True
                        frame_count = 0
                        episode_start_time = time.monotonic()
                        next_frame_time = episode_start_time
                        print(f"\n--- Episode {episode_count} started ---")
                        print("Teach mode active. Move the arm freely.")
                        print("Press Enter to stop and save.")
                    else:
                        recording = False
                        dataset.save_episode()
                        print(f"\nEpisode {episode_count} saved ({frame_count} frames, {frame_count / fps:.1f}s)")
                        episode_count += 1
                        controller.reset_to_home()
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
                    controller.reset_to_home()
                    print(f"Dataset saved to: {dataset.root}")
                    print("Done!")
                    return

            if recording:
                timestamp = time.monotonic() - episode_start_time
                state = get_state_vector(controller.get_joint_state())
                dataset.add_frame({
                    "observation.state": torch.from_numpy(state),
                    "action": torch.from_numpy(state),
                    "task": task,
                })
                dataset.episode_buffer["timestamp"][-1] = timestamp
                frame_count += 1
                if frame_count % fps == 0:
                    print(f"  Recording... {timestamp:.0f}s ({frame_count} frames)", end="\r")
                next_frame_time += dt
                sleep_time = next_frame_time - time.monotonic()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_frame_time = time.monotonic()
            else:
                time.sleep(0.05)  # idle polling


if __name__ == "__main__":
    main()
