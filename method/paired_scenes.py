"""独立研究对照：先固定全部专家准入场景，再让各方法重放同一清单。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from execution_outcomes import classify_execution_error


class SceneReplayError(BaseException):
    """重放不一致不能被官方的换 seed 分支吞掉。"""


def public_fingerprint(observation):
    import numpy as np
    parts = {}
    for name, camera in sorted(observation.get('vision', {}).items()):
        if name not in {'cam_head', 'cam_left_wrist', 'cam_right_wrist'}:
            continue
        if 'color' not in camera:
            continue
        array = np.ascontiguousarray(camera['color'])
        parts['rgb/' + name] = {'shape': list(array.shape), 'dtype': str(array.dtype),
                                'sha256': hashlib.sha256(array.tobytes()).hexdigest()}
    for name, value in sorted(observation.get('state', {}).items()):
        if name not in {'left_arm_joint_state', 'left_ee_joint_state',
                        'right_arm_joint_state', 'right_ee_joint_state'}:
            continue
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
        parts['qpos/' + name] = {'shape': list(array.shape), 'values': array.tolist(),
                                 'sha256': hashlib.sha256(array.tobytes()).hexdigest()}
    if len([key for key in parts if key.startswith('rgb/')]) != 3:
        raise SceneReplayError('initial observation must contain three public RGB cameras')
    if len([key for key in parts if key.startswith('qpos/')]) != 4:
        raise SceneReplayError('initial observation lacks complete public joint/gripper state')
    return parts


def validate_manifest(manifest, config, test_num):
    if manifest.get('status') != 'complete_before_policy':
        raise SceneReplayError('scene admission is incomplete; no automatic reselection')
    for key in ('task', 'condition', 'seed'):
        if manifest.get(key) != config.get(key):
            raise SceneReplayError('scene manifest differs in ' + key)
    all_scenes = manifest.get('scenes', [])
    indices = [int(s['scene_index']) for s in all_scenes]
    if len(set(indices)) != len(indices):
        raise SceneReplayError('scene manifest has duplicate original indices')
    subset = config.get('scene_subset_indices')
    if subset is not None:
        if (not isinstance(subset, list) or any(type(i) is not int for i in subset)
                or len(set(subset)) != len(subset) or not set(subset).issubset(indices)):
            raise SceneReplayError('invalid or unavailable scene subset')
        by_index = {int(s['scene_index']): s for s in all_scenes}
        scenes = [by_index[i] for i in subset]
    else:
        if indices != list(range(len(all_scenes))):
            raise SceneReplayError('full canonical scene indices are not contiguous')
        scenes = all_scenes
    if len(scenes) != test_num or len({int(s['seed']) for s in scenes}) != test_num:
        raise SceneReplayError('scene manifest count/seed uniqueness mismatch')
    if any(not isinstance(s.get('instruction'), str) or not s['instruction'].strip()
           or not s.get('public_initial_fingerprint') for s in scenes):
        raise SceneReplayError('scene instruction or initial observation is missing')
    return scenes


def register_host_manifest(official, config, task_env, args, frequency, write_json, emit):
    canonical_path = Path(config['canonical_scene_manifest_path'])
    canonical = json.loads(canonical_path.read_text())
    selected = validate_manifest(canonical, config, int(config['test_num']))
    scenes = []
    for source in selected:
        index, seed = int(source['scene_index']), int(source['seed'])
        try:
            task_env.setup_demo(now_ep_num=index, seed=seed, is_test=True, **args)
            task_env.set_instruction(instruction=source['instruction'])
            public = official.robotwin_obs_to_xpolicylab(task_env.get_obs(),
                instruction=source['instruction'], env_idx=0, frequency=frequency, task_env=task_env)
            actual = public_fingerprint(public)
            expected = source['public_initial_fingerprint']
            qpos = [key for key in expected if key.startswith('qpos/')]
            if any(actual.get(key) != expected[key] for key in qpos):
                emit('host_registration_qpos_mismatch', seed=seed, expected=expected, actual=actual)
                raise SceneReplayError('cross-host initial public qpos differs; no scene substitution')
            changed = [key for key in expected if key.startswith('rgb/') and actual.get(key) != expected[key]]
            scenes.append({**source, 'original_scene_index': index,
                'canonical_public_initial_fingerprint': expected,
                'public_initial_fingerprint': actual, 'registered_host': config['host_id'],
                'cross_host_rgb_different_fields': changed})
            emit('host_initial_registered', seed=seed, scene_index=index,
                 host_id=config['host_id'], cross_host_rgb_different_fields=changed,
                 qpos_identical=True, policy_calls_so_far=0)
        finally:
            official.safe_close_env(task_env)
    value = {**canonical, 'scenes': scenes, 'status': 'complete_before_policy',
        'canonical_scene_manifest_path': str(canonical_path), 'registered_host': config['host_id'],
        'expert_admission_source': 'canonical_manifest',
        'created_at': time.time(), 'policy_calls_so_far': 0,
        'protocol': 'within_host_exact_initial_pairing_cross_host_RGB_explicitly_recorded'}
    write_json(Path(config['scene_manifest_path']), value)
    return value


def make_paired_evaluator(official, config, emit, write_json):
    manifest_path = Path(config['scene_manifest_path'])

    def evaluate(task_name, task_env, args, model_client, st_seed, *, usr_args,
                 test_num=100, video_size=None, instruction_type=None):
        frequency = int(usr_args.get('frequency', usr_args.get('ctrl_freq', 30)))
        action_type = str(usr_args.get('action_type', 'joint'))
        args['eval_mode'] = True
        if not manifest_path.exists() and config.get('canonical_scene_manifest_path'):
            register_host_manifest(official, config, task_env, args, frequency, write_json, emit)
        if not manifest_path.exists():
            if config.get('scene_subset_indices') is not None:
                raise SceneReplayError('admit full canonical scenes before making subsets')
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            claim_path = manifest_path.with_suffix('.claim.json')
            claim = {'pid': os.getpid(), 'started_at': time.time(), 'task': task_name,
                     'condition': config['condition'], 'seed': config['seed']}
            with claim_path.open('x', encoding='utf-8') as stream:
                json.dump(claim, stream)
            scenes, rejected = [], []
            seed = int(st_seed)
            maximum = int(config.get('max_scene_seed_attempts', test_num * 50))
            admission_args = dict(args)
            admission_args['render_freq'] = 0
            admission_args.pop('eval_video_save_dir', None)
            for attempt in range(maximum):
                index = len(scenes)
                try:
                    task_env.setup_demo(now_ep_num=index, seed=seed, is_test=True, **admission_args)
                    episode_info = task_env.play_once()
                    task_env.close_env()
                    accepted = bool(task_env.plan_success and task_env.check_success())
                    if not accepted:
                        rejected.append({'seed': seed, 'reason': 'expert_rejected'})
                        seed += 1
                        continue
                    # 专家仍关闭UI渲染；登记策略初态时使用与重放完全相同的设置。
                    task_env.setup_demo(now_ep_num=index, seed=seed, is_test=True, **args)
                    instruction = official.build_instruction(args, episode_info, instruction_type,
                                                             test_num, seed)
                    task_env.set_instruction(instruction=instruction)
                    public = official.robotwin_obs_to_xpolicylab(
                        task_env.get_obs(), instruction=instruction, env_idx=0,
                        frequency=frequency, task_env=task_env)
                    fingerprint = public_fingerprint(public)
                    task_env.close_env()
                    scenes.append({'scene_index': index, 'seed': seed, 'instruction': instruction,
                                   'public_initial_fingerprint': fingerprint})
                    emit('scene_admitted', seed=seed, scene_index=index,
                         instruction=instruction, policy_calls_so_far=0)
                except Exception as error:
                    official.safe_close_env(task_env)
                    rejected.append({'seed': seed, 'reason': type(error).__name__, 'error': str(error)})
                    if 'failed to find a rendering device' in str(error).lower():
                        raise SceneReplayError(str(error)) from error
                finally:
                    write_json(manifest_path.with_suffix('.admission.json'), {
                        **claim, 'scenes': scenes, 'rejected': rejected,
                        'policy_calls_so_far': 0, 'updated_at': time.time()})
                seed += 1
                if len(scenes) == test_num:
                    break
            if len(scenes) != test_num:
                raise SceneReplayError('expert-only admission did not fill predeclared scene budget')
            write_json(manifest_path, {**claim, 'status': 'complete_before_policy',
                'policy_calls_so_far': 0, 'scenes': scenes, 'rejected': rejected,
                'created_at': time.time(),
                'protocol': 'research_paired_scene_replay_not_unmodified_official_sampling'})
        manifest = json.loads(manifest_path.read_text())
        scenes = validate_manifest(manifest, config, test_num)
        emit('scene_manifest_ready', path=str(manifest_path), scene_count=len(scenes),
             completed_before_first_policy_call=True)
        if config.get('scene_admission_only'):
            emit('admission_only_complete', policy_calls_so_far=0, scene_count=len(scenes))
            return int(scenes[-1]['seed']) + 1, 0
        task_env.suc = 0
        task_env.test_num = 0
        for local_index, scene in enumerate(scenes):
            seed, index = int(scene['seed']), int(scene['scene_index'])
            try:
                task_env.setup_demo(now_ep_num=index, seed=seed, is_test=True, **args)
                instruction = scene['instruction']
                task_env.set_instruction(instruction=instruction)
                observation = task_env.get_obs()
                public = official.robotwin_obs_to_xpolicylab(
                    observation, instruction=instruction, env_idx=0,
                    frequency=frequency, task_env=task_env)
                actual = public_fingerprint(public)
                if actual != scene['public_initial_fingerprint']:
                    emit('scene_replay_mismatch', seed=seed, expected=scene['public_initial_fingerprint'], actual=actual)
                    raise SceneReplayError('public initial observation differs; no policy was called for this scene')
                emit('scene_replay_verified', seed=seed, scene_index=index, local_episode_index=local_index,
                     instruction=instruction, manifest=str(manifest_path))
            except BaseException:
                official.safe_close_env(task_env)
                raise
            ffmpeg = None
            success = False
            phase = 'video_setup'
            attempt_started = time.time()
            attempt_aborted = False
            try:
                if task_env.eval_video_path is not None:
                    ffmpeg = subprocess.Popen([
                        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
                        '-pixel_format', 'rgb24', '-video_size', video_size,
                        '-framerate', '10', '-i', '-', '-pix_fmt', 'yuv420p',
                        '-vcodec', 'libx264', '-crf', '23',
                        f'{task_env.eval_video_path}/episode{local_index}.mp4'], stdin=subprocess.PIPE)
                    task_env._set_eval_video_ffmpeg(ffmpeg)
                phase = 'policy_rpc'
                official.prepare_policy_case(model_client, task_name, seed, instruction, action_type)
                official.reset_policy(model_client)
                while not official.is_episode_end(task_env):
                    phase = 'observation_conversion'
                    public = official.robotwin_obs_to_xpolicylab(
                        observation, instruction=instruction, env_idx=0,
                        frequency=frequency, task_env=task_env)
                    phase = 'policy_rpc'
                    model_client.call(func_name='update_obs', obs=public)
                    raw_chunk = model_client.call(func_name='get_action')
                    phase = 'policy_output'
                    chunk = official.normalize_action_chunk(raw_chunk)
                    if not chunk:
                        raise RuntimeError('Policy returned an empty action chunk.')
                    for action_index, action in enumerate(chunk):
                        phase = 'action_execution'
                        flat, robotwin_type = official.xpolicylab_action_to_robotwin(
                            action, action_type=action_type, current_observation=observation)
                        task_env.take_action(flat, action_type=robotwin_type)
                        if task_env.eval_success:
                            success = True
                            break
                        if official.is_episode_end(task_env) or action_index + 1 == len(chunk):
                            break
                        observation = task_env.get_obs()
                        public = official.robotwin_obs_to_xpolicylab(
                            observation, instruction=instruction, env_idx=0,
                            frequency=frequency, task_env=task_env)
                        phase = 'policy_rpc'
                        model_client.call(func_name='update_obs', obs=public)
                    if success:
                        break
                    if not official.is_episode_end(task_env):
                        phase = 'observation_conversion'
                        observation = task_env.get_obs()
                phase = 'policy_rpc'
                # 只有正常物理终局才通知策略；API 中断不能写成失败训练记忆。
                official.notify_trial_end(model_client, task_name, seed, success)
            except BaseException as error:
                attempt_aborted = True
                classification = classify_execution_error(error, phase=phase)
                event = ('policy_error' if classification['error_category'] == 'policy_error'
                         else 'infrastructure_error')
                emit(event, seed=seed, error_type=type(error).__name__, phase=phase,
                     error=str(error), traceback=traceback.format_exc(),
                     api_error=classification['error_category'] == 'api_error', **classification)
                emit('episode_outcome', seed=seed, instruction=instruction,
                     episode_index=local_index, original_scene_index=index,
                     attempt_status='aborted', task_success=None, included_in_sr=False,
                     action_steps=getattr(task_env, 'take_action_cnt', None),
                     started_at=attempt_started, ended_at=time.time(), **classification)
                try:
                    official.safe_close_env(task_env)
                except BaseException as cleanup_error:
                    emit('cleanup_error', seed=seed, error_type=type(cleanup_error).__name__,
                         error=str(cleanup_error), traceback=traceback.format_exc())
                raise
            finally:
                if ffmpeg is not None:
                    try:
                        task_env._del_eval_video_ffmpeg()
                    except Exception as cleanup_error:
                        if not attempt_aborted:
                            raise
                        emit('cleanup_error', seed=seed, error_type=type(cleanup_error).__name__,
                             error=str(cleanup_error), traceback=traceback.format_exc())
            emit('episode_outcome', seed=seed, instruction=instruction,
                 episode_index=local_index, original_scene_index=index,
                 attempt_status='completed', task_success=bool(success), included_in_sr=True,
                 action_steps=getattr(task_env, 'take_action_cnt', None),
                 started_at=attempt_started, ended_at=time.time())
            task_env.suc += int(success)
            task_env.test_num = local_index + 1
            task_env.close_env(clear_cache=((local_index + 2) % args['clear_cache_freq'] == 0))
            if task_env.render_freq:
                task_env.viewer.close()
            print(f'Success rate: {task_env.suc}/{task_env.test_num}, current seed: {seed}')
        return int(scenes[-1]['seed']) + 1, task_env.suc

    return evaluate
