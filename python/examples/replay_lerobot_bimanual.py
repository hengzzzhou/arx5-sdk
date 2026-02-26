"""
Replay a bimanual LeRobot dataset on two ARX5 arms.

Usage:
    python replay_lerobot_bimanual.py X5 can0 X5 can1 --root ./data_bimanual
    python replay_lerobot_bimanual.py X5 can0 X5 can1 --root ./data_bimanual --episode 0

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
    """Load bimanual trajectory. Returns ((N, 14) actions, (N,) timestamps, fps)."""
    import json

    with open(os.path.join(root, "meta", "info.json")) as f:
        info = json.load(f)
    fps = info["fps"]

    data_dir = os.path.join(root, "data")
    parquet_files = []
    for dirpath, _, filenames in os.walk(data_dir):
        for fn in sorted(filenames):
            if fn.endswith(".parquet"):
                parquet_files.append(os.path.join(dirpath, fn))

    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    ep_df = df[df["episode_index"] == episode]
    if len(ep_df) == 0:
        available = sorted(df["episode_index"].unique())
        raise ValueError(f"Episode {episode} not found. Available: {available}")

    traj = np.stack(ep_df["action"].values).astype(np.float64)
    timestamps = ep_df["timestamp"].values.astype(np.float64)
    duration = timestamps[-1] - timestamps[0]
    print(f"Loaded episode {episode}: {len(traj)} frames, {duration:.1f}s, dim={traj.shape[1]}")
    return traj, timestamps, fps


def smooth_move_to(left, right, robot_config, traj_first_frame, duration=1.0):
    """Smoothly interpolate both arms to the first trajectory frame."""
    controller_config = left.get_controller_config()
    dt = controller_config.controller_dt
    steps = max(int(duration / dt), 1)

    left_state = left.get_joint_state()
    right_state = right.get_joint_state()
    start_left_pos = left_state.pos().copy()
    start_left_gripper = left_state.gripper_pos
    start_right_pos = right_state.pos().copy()
    start_right_gripper = right_state.gripper_pos

    target_left_pos = traj_first_frame[:6]
    target_left_gripper = traj_first_frame[6]
    target_right_pos = traj_first_frame[7:13]
    target_right_gripper = traj_first_frame[13]

    print(f"  Moving to start position ({duration:.1f}s)...")
    for i in range(1, steps + 1):
        t = i / steps
        alpha = t * t * (3.0 - 2.0 * t)  # smoothstep

        left_cmd = arx5.JointState(robot_config.joint_dof)
        left_cmd.pos()[:] = start_left_pos * (1.0 - alpha) + target_left_pos * alpha
        left_cmd.gripper_pos = float(start_left_gripper * (1.0 - alpha) + target_left_gripper * alpha)
        left.set_joint_cmd(left_cmd)

        right_cmd = arx5.JointState(robot_config.joint_dof)
        right_cmd.pos()[:] = start_right_pos * (1.0 - alpha) + target_right_pos * alpha
        right_cmd.gripper_pos = float(start_right_gripper * (1.0 - alpha) + target_right_gripper * alpha)
        right.set_joint_cmd(right_cmd)

        time.sleep(dt)


def init_controller(model: str, interface: str) -> arx5.Arx5JointController:
    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", robot_config.joint_dof
    )
    controller_config.background_send_recv = True
    return arx5.Arx5JointController(robot_config, controller_config, interface)


@click.command()
@click.argument("left_model")
@click.argument("left_interface")
@click.argument("right_model")
@click.argument("right_interface")
@click.option("--root", required=True, help="LeRobot dataset root directory")
@click.option("--episode", default=0, help="Episode index to replay (default: 0)")
def main(left_model: str, left_interface: str, right_model: str, right_interface: str,
         root: str, episode: int):
    assert left_interface != right_interface, "Left and right arms must be on different CAN interfaces"

    traj, timestamps, fps = load_trajectory(root, episode)
    assert traj.shape[1] == 14, f"Expected 14-dim trajectory (bimanual), got {traj.shape[1]}"

    np.set_printoptions(precision=4, suppress=True)
    print(f"Initializing left arm ({left_model} on {left_interface})...")
    left = init_controller(left_model, left_interface)
    print(f"Initializing right arm ({right_model} on {right_interface})...")
    right = init_controller(right_model, right_interface)
    robot_config = left.get_robot_config()

    left.reset_to_home()
    right.reset_to_home()

    # Replay at controller's native rate for smooth interpolation
    cmd_dt = left.get_controller_config().controller_dt

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
                    smooth_move_to(left, right, robot_config, traj[0], duration=1.0)
                    replaying = True
                    frame_idx = 0
                    replay_start_time = time.monotonic()
                    print("Replay started!")

                elif key_stroke == KeyCode(char="q"):
                    print("Quitting...")
                    left.reset_to_home()
                    right.reset_to_home()
                    return

            if replaying:
                elapsed = time.monotonic() - replay_start_time
                total_duration = timestamps[-1] - timestamps[0]

                if elapsed <= total_duration:
                    # Find which two recorded frames we are between
                    t = elapsed + timestamps[0]
                    # Advance frame_idx to the next frame after current time
                    while frame_idx < len(traj) - 1 and timestamps[frame_idx + 1] <= t:
                        frame_idx += 1

                    if frame_idx >= len(traj) - 1:
                        # At or past the last frame
                        interp = traj[-1]
                    else:
                        # Linearly interpolate between frame_idx and frame_idx+1
                        t0 = timestamps[frame_idx]
                        t1 = timestamps[frame_idx + 1]
                        alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                        alpha = max(0.0, min(1.0, alpha))
                        interp = traj[frame_idx] * (1.0 - alpha) + traj[frame_idx + 1] * alpha

                    left_cmd = arx5.JointState(robot_config.joint_dof)
                    left_cmd.pos()[:] = interp[:6]
                    left_cmd.gripper_pos = float(interp[6])
                    left.set_joint_cmd(left_cmd)

                    right_cmd = arx5.JointState(robot_config.joint_dof)
                    right_cmd.pos()[:] = interp[7:13]
                    right_cmd.gripper_pos = float(interp[13])
                    right.set_joint_cmd(right_cmd)

                    if int(elapsed) > int(elapsed - cmd_dt):
                        print(f"  Replaying... {elapsed:.0f}s / {total_duration:.0f}s", end="\r")
                    time.sleep(cmd_dt)
                else:
                    print(f"\nReplay finished! ({len(traj)} frames, {total_duration:.1f}s)")
                    replaying = False
                    frame_idx = 0
                    left.reset_to_home()
                    right.reset_to_home()
                    print("Press 'r' to replay again, 'q' to quit.")
            else:
                time.sleep(0.05)


if __name__ == "__main__":
    main()
