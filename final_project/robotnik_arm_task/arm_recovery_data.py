"""Actual expert recovery rollouts after small pose/velocity disturbances.

Uses train and validation target groups separately; never reads evaluation runs.
Every trial uses a fresh simulator, replays the recorded expert to a near-target
state, checks reproduction, then physically perturbs joints. The official expert
keeps the saved pre-action reference and actually executes every logged torque.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_SETTINGS = dict(episodes=32, horizon_s=3., stride=10, max_center_distance_m=.04,
    max_center_speed=.12, min_angle_rad=.005, max_angle_rad=.02,
    min_shift_m=.005, max_shift_m=.03, cartesian_speed=.03, max_joint_speed=.15,
    attempts=30, hold_s=.25, tolerance_m=.01)


def recovery_trial(task, episode_path, mode, rng, settings, metadata):
    with np.load(episode_path, allow_pickle=False) as a:
        X0, y0, references = a['X'], a['y'], a['q_ref_before']
        mask = a['nav_done_after'] & (a['ee_error'] <= settings['max_center_distance_m'])
        mask &= np.max(np.abs(X0[:, 11:18]), axis=1) <= settings['max_center_speed']
        centers = np.flatnonzero(mask)
    if not len(centers):
        return None, dict(mode=mode, accepted=False, reason='no near-target center')
    # Choose before the very last sample so this is a correction, not only hold.
    center = int(centers[len(centers)//3])
    target = X0[center, 18:21].copy()
    sim = task.pb.SimInterface('robotnik_arm.json', conf_file_path_ext=str(ROOT), use_gui=False)
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
        hold = np.asarray(sim.GetMotorAngles(0))[4:11].copy()
        names = np.expand_dims(np.asarray(sim.getNameActiveJoints()), 0)
        dyn = task.PinWrapper('robotnik_arm.json', 'pybullet', names, ['pybullet'], False, 0, str(ROOT))
        frame = dyn.pin_model.getFrameId('robot_j2s7s300_end_effector')
        shoulder = dyn.pin_model.getFrameId('robot_j2s7s300_link_2')
        lower, upper = sim.GetBotJointsLimit()
        lower, upper = np.asarray(lower)[4:11], np.asarray(upper)[4:11]
        params = task.ArmControllerParams(**metadata['arm_params'])
        dt = float(sim.GetTimeStep())

        def apply(action):
            cmd = task.MotorCommands()
            cmd.SetControlCmd(np.concatenate((action, np.zeros(6))), ['velocity']*4+['torque']*13)
            sim.Step(cmd, 'torque')

        def fk(qp):
            task.pin.forwardKinematics(dyn.pin_model, dyn.pin_data, qp)
            task.pin.updateFramePlacements(dyn.pin_model, dyn.pin_data)
            return dyn.pin_data.oMf[frame].translation.copy()

        def measure():
            pos, quat = np.asarray(sim.GetBasePosition()), np.asarray(sim.GetBaseOrientation())
            q, qd = np.asarray(sim.GetMotorAngles(0)), np.asarray(sim.GetMotorVelocities(0))
            qp = dyn.ReoderJoints2PinVec(np.concatenate((pos, quat, q)), 'pos')
            ee = fk(qp)
            yaw = task.quaternion2bearing(quat[3], quat[0], quat[1], quat[2])
            return np.concatenate((pos, [yaw], q[4:11], qd[4:11], target)), qp, ee

        for k in range(center):
            apply(y0[k])
        X, qp, original_ee = measure()
        replay_error = float(np.max(np.abs(X-X0[center])))
        if replay_error > 1e-6:
            raise ValueError(f'Expert replay mismatch in {episode_path.name}: {replay_error:g}')
        np.testing.assert_allclose(qp[7:14], X[4:11], rtol=0, atol=1e-10)
        reference = references[center].copy()
        nominal, _, _ = task.arm_controller(dyn, qp, X[4:11], X[11:18], reference.copy(), hold,
            target, original_ee.copy(), frame, shoulder, lower, upper, dt, params, reach_active=True)
        np.testing.assert_allclose(nominal, y0[center, 4:11], rtol=0, atol=1e-6)
        limited = lower < upper
        room = ~limited | ((X[4:11]-settings['max_angle_rad'] >= lower+params.limit_margin) &
                           (X[4:11]+settings['max_angle_rad'] <= upper-params.limit_margin))
        joints = np.flatnonzero(room)
        if len(joints) < 2:
            return None, dict(mode=mode, accepted=False, reason='insufficient joint-limit margin')
        selected = None
        for _ in range(settings['attempts']):
            offset = np.zeros(7)
            chosen = rng.choice(joints, size=int(rng.integers(2, min(3,len(joints))+1)), replace=False)
            offset[chosen] = rng.uniform(settings['min_angle_rad'], settings['max_angle_rad'], len(chosen))*rng.choice([-1,1], len(chosen))
            perturbed = qp.copy(); perturbed[7:14] += offset
            ee = fk(perturbed)
            shift = float(np.linalg.norm(ee-original_ee))
            if settings['min_shift_m'] <= shift <= settings['max_shift_m']:
                selected = perturbed, ee, shift
                break
        if selected is None:
            return None, dict(mode=mode, accepted=False, reason='no valid Cartesian shift')
        perturbed, ee, shift = selected
        task.pin.computeJointJacobians(dyn.pin_model, dyn.pin_data, perturbed)
        J = task.pin.computeFrameJacobian(dyn.pin_model, dyn.pin_data, perturbed, frame,
                                          task.pin.LOCAL_WORLD_ALIGNED)[:3, 6:13]
        direction = target-ee
        velocity = np.zeros(7)
        if mode != 'stationary' and np.linalg.norm(direction)>1e-8:
            desired = settings['cartesian_speed']*direction/np.linalg.norm(direction)
            velocity = J.T@np.linalg.solve(J@J.T+.02**2*np.eye(3), desired)
            velocity *= 1 if mode=='toward' else -1
            velocity *= min(1., settings['max_joint_speed']/max(float(np.max(np.abs(velocity))), 1e-12))
        # Convert measured joint coordinates back to Bullet's raw coordinates.
        bot = sim.bot[0]
        motor_direction, motor_offset = np.asarray(bot.motor_direction), np.asarray(bot.motor_offset)
        for j in range(7):
            index = 4+j
            c.resetJointState(rid, bot.active_joint_ids[index],
                targetValue=float(perturbed[7+j]/motor_direction[index]+motor_offset[index]),
                targetVelocity=float(velocity[j]/motor_direction[index]))
        sim.ReceiveObservation()
        start_X, _, _ = measure()
        np.testing.assert_allclose(start_X[4:11], perturbed[7:14], rtol=0, atol=1e-10)
        np.testing.assert_allclose(start_X[11:18], velocity, rtol=0, atol=1e-10)
        xs, ys, errors, times, speeds = [], [], [], [], []
        held = longest = 0
        for k in range(int(np.ceil(settings['horizon_s']/dt))+1):
            state, state_qp, ee = measure()
            if not np.isfinite(state).all():
                raise RuntimeError('Non-finite expert recovery state')
            error = float(np.linalg.norm(target-ee)); speed = float(np.max(np.abs(state[11:18])))
            errors.append(error); times.append(k*dt); speeds.append(speed)
            held = held+1 if error<settings['tolerance_m'] and speed<.03 else 0
            longest = max(longest, held)
            if k == int(np.ceil(settings['horizon_s']/dt)):
                break
            torque, reference, _ = task.arm_controller(dyn, state_qp, state[4:11], state[11:18],
                reference, hold, target, ee.copy(), frame, shoulder, lower, upper, dt, params, reach_active=True)
            if k % settings['stride']==0:
                xs.append(state.copy()); ys.append(torque.copy())
            apply(np.concatenate((np.zeros(4), torque)))
        # Require a settled FINAL interval, not merely a transient earlier hit.
        recovered = max(0,held-1)*dt >= settings['hold_s']-1e-12
        summary = dict(mode=mode, accepted=bool(recovered), center_row=center,
            replay_max_error=replay_error, initial_shift_m=shift,
            closest_distance_m=min(errors), final_distance_m=errors[-1],
            longest_settled_hold_s=max(0,longest-1)*dt,
            final_settled_hold_s=max(0,held-1)*dt,
            time_s=times[::settings['stride']], ee_distance_m=errors[::settings['stride']],
            max_joint_speed=speeds[::settings['stride']])
        return (dict(X=np.asarray(xs), y=np.asarray(ys)) if recovered else None), summary
    finally:
        sim.pybullet_client.disconnect()


def load_or_build_recovery_pool(dataset, episode_ids, split, settings=None, seed=45, cache_root=None):
    from arm_angle_augmentation import quiet_replay
    settings = dict(DEFAULT_SETTINGS if settings is None else settings)
    dataset = Path(dataset)
    if split not in (0,1):
        raise ValueError('Recovery data may use train or validation episodes, never test')
    if (settings['episodes']<1 or settings['stride']<1 or settings['attempts']<1 or
            any(not np.isfinite(v) or v<=0 for v in settings.values()) or
            settings['min_shift_m']>=settings['max_shift_m'] or
            settings['min_angle_rad']>settings['max_angle_rad']):
        raise ValueError('Invalid recovery settings')
    allowed = set(map(int,episode_ids))
    manifest = json.loads((dataset.parent/'episodes.json').read_text())
    candidates = sorted(e['episode_id'] for e in manifest if e['episode_id'] in allowed and e['split']==split and e['success'])
    if not candidates:
        raise ValueError('No successful episodes in requested recovery split')
    rng = np.random.default_rng(seed)
    selected = sorted(rng.choice(candidates, size=min(settings['episodes'],len(candidates)), replace=False).tolist())
    metadata = json.loads((dataset.parent/'metadata.json').read_text())
    hashes = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
              ('arm_recovery_data.py','robotnik_arm_task.py','robotnik_arm_control.py','configs/robotnik_arm.json')}
    if hashes['robotnik_arm_control.py'] != metadata['source_sha256']['robotnik_arm_control.py']:
        raise ValueError('Expert controller no longer matches dataset')
    info = dict(dataset=str(dataset.resolve()), dataset_bytes=dataset.stat().st_size,
        dataset_mtime_ns=dataset.stat().st_mtime_ns, source_sha256=hashes,
        metadata_sha256=hashlib.sha256((dataset.parent/'metadata.json').read_bytes()).hexdigest(),
        split=split, episodes=selected, settings=settings, seed=seed,
        method='Physical multi-joint pose/velocity disturbance; actual official expert recovery with source pre-action reference. Stationary/toward/away initial velocity. Keep trajectories that settle within 1 cm for 0.25 s. No evaluation targets used.')
    key = hashlib.sha256(json.dumps(info,sort_keys=True).encode()).hexdigest()[:20]
    cache = (Path(cache_root) if cache_root is not None else ROOT/'data'/'augmentation_cache')/f'arm_recovery_{key}.npz'
    if cache.exists():
        with np.load(cache,allow_pickle=False) as a:
            pool = {k:a[k] for k in ('X','y','parent_episode')}
            info = json.loads(str(a['info'].item()))
        validate(pool,allowed)
        print(f'Recovery cache: {cache} ({len(pool["X"])} rows)',flush=True)
        return pool,info
    import collect_expert_data
    task = collect_expert_data.task
    xs,ys,parents,summaries = [],[],[],[]
    for number,eid in enumerate(selected,1):
        for mode in ('stationary','toward','away'):
            print(f'Recovery split={split}: episode {number}/{len(selected)}, id={eid}, {mode}',flush=True)
            path = dataset.parent/f'episode_{eid:04d}.npz'
            with np.load(path,allow_pickle=False) as a:
                if not np.all(a['split']==split) or not np.all(a['episode_success']):
                    raise ValueError('Recovery split mismatch')
            with quiet_replay():
                trial,summary = recovery_trial(task,path,mode,rng,settings,metadata)
            summary['episode_id']=eid; summaries.append(summary)
            if trial is not None:
                xs.extend(trial['X']);ys.extend(trial['y']);parents.extend([eid]*len(trial['X']))
    pool=dict(X=np.asarray(xs),y=np.asarray(ys),parent_episode=np.asarray(parents,dtype=np.int64))
    validate(pool,allowed)
    info.update(cache=str(cache),samples=len(xs),trials=summaries)
    cache.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache.parent,suffix='.npz',delete=False) as f:
        temp=Path(f.name)
    try:
        np.savez_compressed(temp,**pool,info=np.asarray(json.dumps(info)))
        temp.replace(cache)
    finally:
        temp.unlink(missing_ok=True)
    return pool,info


def validate(pool,allowed):
    n=len(pool['X'])
    if not n or pool['X'].shape!=(n,21) or pool['y'].shape!=(n,7) or pool['parent_episode'].shape!=(n,):
        raise ValueError('Empty/invalid recovered trajectory pool; inspect expert recovery settings')
    if not np.isfinite(pool['X']).all() or not np.isfinite(pool['y']).all():
        raise ValueError('Non-finite recovery dataset')
    if not set(pool['parent_episode'].tolist())<=allowed:
        raise ValueError('Recovery pool crosses target split')
