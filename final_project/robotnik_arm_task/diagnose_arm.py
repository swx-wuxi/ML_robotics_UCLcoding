"""Generate only four expert/MLP comparison PNGs for the recorded episode.

RoboEnv/.pixi/envs/default/bin/python final_project/robotnik_arm_task/compare_episode_arm.py
Default duration matches the recorded arm phase; --seconds overrides it.
Navigation replays recorded commands. Original NPZ lacks the full hidden physics
state; reconstructed full start states are checked equal between fresh worlds.
"""
import os
os.environ.setdefault('ROBOENV_HEADLESS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/robotnik_matplotlib')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import contextlib
import copy
import ctypes
import sys
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

@contextlib.contextmanager
def quiet_native(stderr=False):
    """Silence Python and C simulator chatter without creating log files."""
    sys.stdout.flush(); sys.stderr.flush()
    descriptors = (1, 2) if stderr else (1,)
    saved = {fd:os.dup(fd) for fd in descriptors}
    try:
        with open(os.devnull, 'w') as sink:
            for fd in descriptors:
                os.dup2(sink.fileno(), fd)
            yield
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        ctypes.CDLL(None).fflush(None)
        for fd, original in saved.items():
            os.dup2(original, fd); os.close(original)


with quiet_native(stderr=True):
    import robotnik_arm_task as task

ROOT = Path(__file__).resolve().parent
DEFAULT_EPISODE = ROOT/'evaluation/final_20261027T171552_315325Z/episode_0001.npz'

def predict(model, X):
    h = (np.asarray(X, dtype=np.float32)-model['nn_x_mean'])/model['nn_x_scale']
    for i in range(3):
        h = h@model[f'nn_w{i}']+model[f'nn_b{i}']
        if i < 2:
            h = np.maximum(h, 0)
    return h*model['nn_y_scale']+model['nn_y_mean']

def rollout(branch, source, settings, model, metadata, seconds):
    start = int(np.flatnonzero(source['control_phase']=='arm')[0])
    target = source['target']
    sim = task.pb.SimInterface('robotnik_arm.json', conf_file_path_ext=str(ROOT), use_gui=False)
    snapshot = None
    try:
        sim.SetFloorFriction(1.)
        c, rid = sim.pybullet_client, sim.bot[0].bot_pybullet
        for j in range(c.getNumJoints(rid)):
            if 'wheel' in c.getJointInfo(rid, j)[1].decode():
                c.changeDynamics(rid, j, contactStiffness=30000, contactDamping=1000,
                                 anisotropicFriction=[1., .01, 1.])
        sim.bot[0].servo_motor_model.set_motor_gains(400, 30)
        task.configure_gripper_fingers(sim, c, rid, task.FIXED_FINGER_ANGLES)
        visual = c.createVisualShape(task.p.GEOM_SPHERE, radius=.03, rgbaColor=[1, 0, 0, .85])
        c.createMultiBody(baseMass=0, baseVisualShapeIndex=visual, basePosition=target.tolist())
        names = np.expand_dims(np.asarray(sim.getNameActiveJoints()), 0)
        dyn = task.PinWrapper('robotnik_arm.json', 'pybullet', names, ['pybullet'], False, 0, str(ROOT))
        frame = dyn.pin_model.getFrameId('robot_j2s7s300_end_effector')
        shoulder = dyn.pin_model.getFrameId('robot_j2s7s300_link_2')
        lower, upper = sim.GetBotJointsLimit()
        lower, upper = np.asarray(lower)[4:11], np.asarray(upper)[4:11]
        dt = float(sim.GetTimeStep())
        np.testing.assert_allclose(dt, np.median(np.diff(source['time'])), rtol=0, atol=1e-10)
        ap = task.ArmControllerParams(**metadata['arm_params'])
        hold = np.asarray(sim.GetMotorAngles(0))[4:11].copy()

        def measure():
            pos, quat = np.asarray(sim.GetBasePosition()), np.asarray(sim.GetBaseOrientation())
            q, qd = np.asarray(sim.GetMotorAngles(0)), np.asarray(sim.GetMotorVelocities(0))
            qp = dyn.ReoderJoints2PinVec(np.concatenate((pos, quat, q)), 'pos')
            task.pin.forwardKinematics(dyn.pin_model, dyn.pin_data, qp)
            task.pin.updateFramePlacements(dyn.pin_model, dyn.pin_data)
            ee = dyn.pin_data.oMf[frame].translation.copy()
            yaw = task.quaternion2bearing(quat[3], quat[0], quat[1], quat[2])
            return np.concatenate((pos, [yaw], q[4:11], qd[4:11], target)), ee, qp

        def apply(action):
            cmd = task.MotorCommands()
            cmd.SetControlCmd(np.concatenate((action, np.zeros(6))), ['velocity']*4+['torque']*13)
            sim.Step(cmd, 'torque')

        for k in range(start):
            apply(source['actions'][k])
        initial = measure()[0]
        np.testing.assert_allclose(initial, source['X'][start], rtol=0, atol=1e-10)
        snapshot = c.saveState()
        bots = copy.deepcopy(sim.bot)
        fields = {key: copy.deepcopy(getattr(sim, key)) for key in
                  ('step_counter', 'observation_history', 'last_action', 'last_action_time')
                  if hasattr(sim, key)}
        # The expert held initial posture throughout navigation, so its internal
        # reference starts at hold, not the first post-update shadow reference.
        reference = hold.copy()
        c.restoreState(stateId=snapshot)
        sim.bot = copy.deepcopy(bots)
        for key, value in fields.items():
            setattr(sim, key, copy.deepcopy(value))
        np.testing.assert_array_equal(measure()[0], initial)
        pos, quat = c.getBasePositionAndOrientation(rid)
        lin, ang = c.getBaseVelocity(rid)
        joints = c.getJointStates(rid, list(range(c.getNumJoints(rid))))
        physical = np.concatenate((pos, quat, lin, ang,
                                   np.asarray([(j[0], j[1]) for j in joints]).ravel()))
        count = len(source['X'])-start if seconds is None else int(np.ceil(seconds/dt))
        rows, torques = [], []
        for k in range(count+1):
            X, ee, qp = measure()
            if not np.isfinite(X).all() or not np.isfinite(ee).all():
                raise RuntimeError('Non-finite simulation state')
            rows.append((X.copy(), ee.copy()))
            if k == count:
                break
            if branch == 'expert':
                torque, reference, _ = task.arm_controller(dyn, qp, X[4:11], X[11:18], reference,
                    hold, target, ee.copy(), frame, shoulder, lower, upper, dt, ap, reach_active=True)
            else:
                torque = predict(model, X)
            if not np.isfinite(torque).all():
                raise RuntimeError('Non-finite torque')
            if settings['clip_commands']:
                torque = np.clip(torque, -ap.max_torque, ap.max_torque)
            torques.append(torque.copy())
            apply(np.concatenate((np.zeros(4), torque)))
        X, ee = map(np.asarray, zip(*rows))
        return dict(time=np.arange(count+1)*dt, X=X, ee=ee, torque=np.asarray(torques), initial=physical)
    finally:
        if snapshot is not None:
            sim.pybullet_client.removeState(snapshot)
        sim.pybullet_client.disconnect()


def plots(traces, target, out):
    colors = dict(expert='tab:blue', mlp='tab:orange')
    fig, axes = plt.subplots(7, 2, figsize=(12, 16), sharex=True)
    for j in range(7):
        for name, d in traces.items():
            axes[j, 0].plot(d['time'], d['X'][:, 4+j], label=name, color=colors[name])
            axes[j, 1].plot(d['time'], d['X'][:, 11+j], label=name, color=colors[name])
        axes[j, 0].set_ylabel(f'Joint {j+1}')
    axes[0, 0].set_title('Angle (rad)'); axes[0, 1].set_title('Velocity (rad/s)')
    axes[0, 0].legend(); axes[0, 1].legend()
    for ax in axes.flat:
        ax.grid(alpha=.3)
    for ax in axes[-1]:
        ax.set_xlabel('Time since arm start (s)')
    fig.tight_layout(); fig.savefig(out/'joint_angle_velocity.png', dpi=150); plt.close(fig)

    fig, axes = plt.subplots(7, 1, figsize=(11, 15), sharex=True)
    for j, ax in enumerate(axes):
        for name, d in traces.items():
            ax.plot(d['time'][:-1], d['torque'][:, j], label=name, color=colors[name])
        ax.set_ylabel(f'J{j+1} (N m)'); ax.grid(alpha=.3)
    axes[0].legend(); axes[0].set_title('Applied torque commands on independent rollouts')
    axes[-1].set_xlabel('Time since arm start (s)')
    fig.tight_layout(); fig.savefig(out/'joint_torque.png', dpi=150); plt.close(fig)

    fig = plt.figure(figsize=(9, 8)); ax = fig.add_subplot(111, projection='3d')
    for name, d in traces.items():
        ax.plot(*d['ee'].T, label=name, color=colors[name])
        ax.scatter(*d['ee'][-1], color=colors[name], marker='x', s=60)
    ax.scatter(*target, marker='*', color='red', s=160, label='Target')
    ax.scatter(*traces['expert']['ee'][0], color='black', label='Shared start')
    points = np.vstack((target[None], *(d['ee'] for d in traces.values())))
    center = (points.max(0)+points.min(0))/2
    radius = max(float(np.ptp(points, axis=0).max())/2, .01)
    ax.set_xlim(center[0]-radius, center[0]+radius)
    ax.set_ylim(center[1]-radius, center[1]+radius)
    ax.set_zlim(center[2]-radius, center[2]+radius)
    ax.set_box_aspect((1, 1, 1)); ax.legend()
    ax.set(xlabel='World X (m)', ylabel='World Y (m)', zlabel='World Z (m)')
    fig.tight_layout(); fig.savefig(out/'ee_path_3d.png', dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for name, d in traces.items():
        error = np.linalg.norm(d['ee']-target, axis=1)*100
        for ax in axes:
            ax.plot(d['time'], error, label=name, color=colors[name])
    for ax in axes:
        ax.axhline(1, color='green', linestyle='--', label='1 cm')
        ax.set_ylabel('EE distance (cm)'); ax.grid(alpha=.3); ax.legend()
    axes[1].set_ylim(0, 15); axes[1].set_title('Distance zoom: 0–15 cm')
    axes[-1].set_xlabel('Time since arm start (s)')
    fig.tight_layout(); fig.savefig(out/'ee_distance.png', dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, default=DEFAULT_EPISODE)
    parser.add_argument('--seconds', type=float)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.seconds is not None and (not np.isfinite(args.seconds) or args.seconds <= 0):
        parser.error('seconds must be finite and positive')
    settings = json.loads((args.episode.parent/'settings.json').read_text())
    model_path = Path(settings['model'])
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != settings['model_sha256']:
        raise ValueError('Original model hash mismatch')
    with np.load(model_path, allow_pickle=False) as a:
        model = {k:a[k] for k in a.files}
    if int(model.get('arm_window_steps', 1)) != 1 or len(model['nn_x_mean']) != 21:
        raise ValueError('This plot script expects the original single-frame MLP')
    metadata = json.loads((Path(str(model['dataset_path'].item())).parent/'metadata.json').read_text())
    with np.load(args.episode, allow_pickle=False) as a:
        source = {k:a[k] for k in a.files}
    traces = {name:rollout(name, source, settings, model, metadata, args.seconds) for name in ('expert', 'mlp')}
    np.testing.assert_array_equal(traces['expert']['initial'], traces['mlp']['initial'])
    np.testing.assert_array_equal(traces['expert']['X'][0], traces['mlp']['X'][0])
    out = args.output or ROOT/'diagnostics'/datetime.now(timezone.utc).strftime('arm_plots_%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True, exist_ok=False)
    plots(traces, source['target'], out)


if __name__ == '__main__':
    # Suppress simulator chatter; only PNG files are produced. Exceptions remain
    # visible on stderr, and no log/report/NPZ files are created.
    with quiet_native():
        main()
