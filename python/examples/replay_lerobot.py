"""
Replay a LeRobot dataset on the ARX5 arm.

Usage:
    python replay_lerobot.py X5 can0 --root /home/ubuntu/code/robotic/arx5-sdk/data
    python replay_lerobot.py X5 can0 --root ./data --episode 0

Controls:
    r - Start replay
    q - Quit
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import click

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import arx5_interface as arx5
from peripherals.keystroke_counter import KeystrokeCounter, KeyCode


def load_trajectory(root: str, episode: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Load joint trajectory from a LeRobot dataset.

    Returns:
        (trajectory, timestamps, fps) where trajectory is (N, 7) array.
    """
    import json

    with open(os.path.join(root, "meta", "info.json")) as f:
        info = json.load(f)
    fps = info["fps"]

    # Read all parquet data files
    data_dir = os.path.join(root, "data")
    parquet_files = []
    for dirpath, _, filenames in os.walk(data_dir):
        for fn in sorted(filenames):
            if fn.endswith(".parquet"):
                parquet_files.append(os.path.join(dirpath, fn))

    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

    # Filter to requested episode
    ep_df = df[df["episode_index"] == episode]
    if len(ep_df) == 0:
        available = sorted(df["episode_index"].unique())
        raise ValueError(f"Episode {episode} not found. Available: {available}")

    traj = np.stack(ep_df["action"].values).astype(np.float64)
    timestamps = ep_df["timestamp"].values.astype(np.float64)
    duration = timestamps[-1] - timestamps[0]
    print(f"Loaded episode {episode}: {len(traj)} frames, {duration:.1f}s")
    return traj, timestamps, fps


def smooth_move_to(controller, robot_config, target_pos, target_gripper, duration=1.0):
    """Smoothly interpolate from current position to target over duration seconds."""
    controller_config = controller.get_controller_config()
    dt = controller_config.controller_dt
    steps = max(int(duration / dt), 1)

    cur_state = controller.get_joint_state()
    start_pos = cur_state.pos().copy()
    start_gripper = cur_state.gripper_pos

    print(f"  Moving to start position ({duration:.1f}s)...")
    for i in range(1, steps + 1):
        # Smooth ease-in-out
        t = i / steps
        alpha = t * t * (3.0 - 2.0 * t)  # smoothstep

        cmd = arx5.JointState(robot_config.joint_dof)
        cmd.pos()[:] = start_pos * (1.0 - alpha) + target_pos * alpha
        cmd.gripper_pos = float(start_gripper * (1.0 - alpha) + target_gripper * alpha)
        controller.set_joint_cmd(cmd)
        time.sleep(dt)


@click.command()
@click.argument("model")  # ARX arm model: X5 or L5
@click.argument("interface")  # CAN bus name (can0 etc.)
@click.option("--root", required=True, help="LeRobot dataset root directory")
@click.option("--episode", default=0, help="Episode index to replay (default: 0)")
def main(model: str, interface: str, root: str, episode: int):
    # Load trajectory
    traj, timestamps, fps = load_trajectory(root, episode)

    # Initialize robot
    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", robot_config.joint_dof
    )
    controller_config.background_send_recv = True
    controller = arx5.Arx5JointController(robot_config, controller_config, interface)
    np.set_printoptions(precision=4, suppress=True)

    controller.reset_to_home()

    cmd_dt = controller.get_controller_config().controller_dt

    print("\nPress 'r' to start replay, 'q' to quit.")

    with KeystrokeCounter() as key_counter:
        replaying = False
        frame_idx = 0

        while True:
            press_events = key_counter.get_press_events()
            for key_stroke in press_events:
                if key_stroke == KeyCode(char="r"):
                    if replaying:
                        print("Already replaying!")
                        continue
                    # Smoothly move to the first frame before replaying
                    smooth_move_to(
                        controller, robot_config,
                        target_pos=traj[0, :6],
                        target_gripper=traj[0, 6],
                        duration=1.0,
                    )
                    replaying = True
                    frame_idx = 0
                    replay_start_time = time.monotonic()
                    print("Replay started!")

                elif key_stroke == KeyCode(char="q"):
                    print("Quitting...")
                    controller.reset_to_home()
                    return

            if replaying:
                elapsed = time.monotonic() - replay_start_time
                total_duration = timestamps[-1] - timestamps[0]

                if elapsed <= total_duration:
                    t = elapsed + timestamps[0]
                    while frame_idx < len(traj) - 1 and timestamps[frame_idx + 1] <= t:
                        frame_idx += 1

                    if frame_idx >= len(traj) - 1:
                        interp = traj[-1]
                    else:
                        t0 = timestamps[frame_idx]
                        t1 = timestamps[frame_idx + 1]
                        alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                        alpha = max(0.0, min(1.0, alpha))
                        interp = traj[frame_idx] * (1.0 - alpha) + traj[frame_idx + 1] * alpha

                    cmd = arx5.JointState(robot_config.joint_dof)
                    cmd.pos()[:] = interp[:6]
                    cmd.gripper_pos = float(interp[6])
                    controller.set_joint_cmd(cmd)

                    if int(elapsed) > int(elapsed - cmd_dt):
                        print(f"  Replaying... {elapsed:.0f}s / {total_duration:.0f}s", end="\r")
                    time.sleep(cmd_dt)
                else:
                    print(f"\nReplay finished! ({len(traj)} frames, {total_duration:.1f}s)")
                    replaying = False
                    frame_idx = 0
                    controller.reset_to_home()
                    print("Press 'r' to replay again, 'q' to quit.")
            else:
                time.sleep(0.05)


if __name__ == "__main__":
    main()
