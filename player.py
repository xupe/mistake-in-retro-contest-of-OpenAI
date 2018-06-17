from abc import ABC, abstractmethod
from collections import OrderedDict
import time

class Player(ABC):
    @abstractmethod
    def play(self):
        pass


class BasicPlayer(Player):
    def __init__(self, env, model, batch_size=1):
        self.env = env
        self.model = model
        self.batch_size = batch_size
        self._needs_reset = True
        self._cur_state = None
        self._last_obs = None
        self._episode_id = -1
        self._episode_step = 0
        self._total_reward = 0.0


    def play(self):
        return [self._gather_transition() for _ in range(self.batch_size)]


    def _gather_transition(self):
        if self._needs_reset:
            self._needs_reset = False
            self._cur_state = self.model.start_state(1)
            self._last_obs = self.env.reset()
            self._episode_id += 1
            self._episode_step = 0
            self._total_reward = 0.0
        output = self.model.step([self._last_obs], self._cur_state)
        new_obs, rew, self._needs_reset, info = self.env.step(output['actions'][0])
        self._total_reward += rew
        res = {
            'obs': self._last_obs,
            'model_outs': output,
            'rewards': [rew],
            'new_obs': (new_obs if not self._needs_reset else None),
            'info': info,
            'start_state': self._cur_state,
            'episode_id': self._episode_id,
            'episode_step': self._episode_step,
            'end_time': time.time(),
            'is_last': self._needs_reset,
            'total_reward': self._total_reward
        }
        self._cur_state = output['states']
        self._last_obs = new_obs
        self._episode_step += 1
        return res


class NStepPlayer(Player):
    def __init__(self, player, num_steps):
        self.player = player
        self.num_steps = num_steps
        self._ep_to_history = OrderedDict()

    def play(self):
        while True:
            transes = self._play_once()
            if transes:
                return transes

    def _play_once(self):
        for trans in self.player.play():
            assert len(trans['rewards']) == 1
            ep_id = trans['episode_id']
            if ep_id in self._ep_to_history:
                self._ep_to_history[ep_id].append(trans)
            else:
                self._ep_to_history[ep_id] = [trans]

        res = []
        for ep_id, history in list(self._ep_to_history.items()):
            while history:
                trans = self._next_transition(history)
                if trans is None:
                    break
                res.append(trans)
            if not history:
                del self._ep_to_history[ep_id]
        return res

    def _next_transition(self, history):
        if len(history)  < self.num_steps:
            if not history[-1]['is_last']:
                return None

        res = history[0].copy()
        res['rewards'] = [h['rewards'][0] for h in history[:self.num_steps]]
        res['total_reward'] += sum(h['rewards'][0] for h in history[:self.num_steps])
        if len(history) >= self.num_steps:
            res['new_obs'] = history[self.num_steps-1]['new_obs']
        else:
            res['new_obs'] = None
        del history[0]
        return res

class BatchedPlayer(Player):
    def __init__(self, batched_env, model, num_timesteps=1):
        self.batched_env = batched_env
        self.model = model
        self.num_timesteps = num_timesteps
        self._cur_states = None
        self._last_obses = None
        self._episode_ids = [list(range(start, start+batched_env.num_envs_per_sub_batches))
                             for start in range(0, batched_env.num_envs,
                                                batched_env.num_envs_per_sub_batches)]
        self._episode_steps = [[0] * batched_env.num_envs_per_sub_batches
                               for _ in range(batched_env.num_sub_batches)]
        self._next_episode_id = batched_env.num_envs
        self._total_rewards = [[0.0] * batched_env.num_envs_per_sub_batches
                               for _ in range(batched_env.num_sub_batches)]

    def play(self):
        if self._cur_states is None:
            self._setup()
        results = []
        for _ in range(self.num_timesteps):
            for i in range(self.batched_env.num_sub_batches):
                results.extend(self._step_sub_batch(i))
        return results

    def _step_sub_batch(self, sub_batch):
        model_outs = self.model.step(self._last_obses[sub_batch], self._cur_states[sub_batch])
        self.batched_env.step_start(model_outs['actions'], sub_batch=sub_batch)
        outs = self.batched_env.step_wait(sub_batch=sub_batch)
        end_time = time.time()
        transitions = []
        for i, (obs, rew, done, info) in enumerate(zip(*outs)):
            self._total_rewards[sub_batch][i] += rew
            transitions.append({
                'obs': self._last_obses[sub_batch][i],
                'model_outs': _reduce_model_outs(model_outs, i),
                'rewards': [rew],
                'new_obs': (obs if not done else None),
                'info': info,
                'start_state': _reduce_states(self._cur_states[sub_batch], i),
                'episode_id': self._episode_ids[sub_batch][i],
                'episode_step': self._episode_steps[sub_batch][i],
                'end_time': end_time,
                'is_last': done,
                'total_reward': self._total_rewards[sub_batch][i]
            })
            if done:
                _inject_state(model_outs['states'], self.model.start_state(1), i)
                self._episode_ids[sub_batch][i] = self._next_episode_id
                self._next_episode_id += 1
                self._episode_steps[sub_batch][i] = 0
                self._total_rewards[sub_batch][i] = 0.0
            else:
                self._episode_steps[sub_batch][i] += 1
        self._cur_states[sub_batch] = model_outs['states']
        self._last_obses[sub_batch] = outs[0]
        return transitions


    def _setup(self):
        self._cur_states = []
        self._last_obses = []
        for i in range(self.batched_env.num_sub_batches):
            self._cur_states.append(self.model.start_state(self.batched_env.num_envs_per_sub_batches))
            self.batched_env.reset_start(sub_batch=i)
            self._last_obses.append(self.batched_env.reset_wait(sub_batch=i))


def _reduce_states(state_batch, env_idx):
    """
    Reduce a batch of states to a batch of one state.
    """
    if state_batch is None:
        return None
    elif isinstance(state_batch, tuple):
        return tuple(_reduce_states(s, env_idx) for s in state_batch)
    return state_batch[env_idx : env_idx+1].copy()

def _inject_state(state_batch, state, env_idx):
    """
    Replace the state at the given index with a new state.
    """
    if state_batch is None:
        return
    elif isinstance(state_batch, tuple):
        return tuple(_inject_state(sb, s, env_idx)
                     for sb, s in zip(state_batch, state))
    state_batch[env_idx : env_idx+1] = state

def _reduce_model_outs(model_outs, env_idx):
    """
    Reduce a batch of model outputs to a batch of one
    model output.
    """
    out = dict()
    for key in model_outs:
        val = model_outs[key]
        if val is None:
            out[key] = None
        elif isinstance(val, tuple):
            out[key] = _reduce_states(val, env_idx)
        else:
            out[key] = val[env_idx : env_idx+1].copy()
    return out


