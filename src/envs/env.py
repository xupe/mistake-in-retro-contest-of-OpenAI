from abc import ABC, abstractmethod, abstractproperty

import gym
import numpy as np
import sys
sys.path.append("..")

from spaces.gym import StackedBoxSpace

class BatchedEnv(ABC):
    @property
    def num_envs(self):
        return self.num_sub_batches * self.num_envs_per_sub_batches

    @abstractproperty
    def num_sub_batches(self):
        pass

    @abstractproperty
    def num_envs_per_sub_batches(self):
        pass

    @abstractmethod
    def reset_start(self, sub_batch=0):
        pass

    @abstractmethod
    def reset_wait(self, sub_batch=0):
        pass

    @abstractmethod
    def step_start(self, actions, sub_batch=0):
        pass

    @abstractmethod
    def step_wait(self, sub_batch=0):
        pass

    @abstractmethod
    def close(self):
        pass

class BatchedGymEnv(BatchedEnv):
    def __init__(self, envs):
        self.action_space = envs[0][0].action_space
        self.envs = envs
        self.observation_space = envs[0][0].observation_space
        self._step_actions = [None] * len(self.envs)
        self._states = [None] * len(self.envs)

    @property
    def num_sub_batches(self):
        return len(self.envs)

    @property
    def num_envs_per_sub_batches(self):
        return len(self.envs[0])

    def reset_start(self, sub_batch=0):
        assert self._states[sub_batch] is None
        self._states[sub_batch] = 'reset'

    def reset_wait(self, sub_batch=0):
        assert self._states[sub_batch] == 'reset'
        self._states[sub_batch] = None
        return [env.reset() for env in self.envs[sub_batch]]

    def step_start(self, actions, sub_batch=0):
        assert len(actions) == self.num_envs_per_sub_batches
        assert self._states[sub_batch] is None
        self._states[sub_batch] = 'step'
        self._step_actions[sub_batch] = actions

    def step_wait(self, sub_batch=0):
        assert self._step_actions[sub_batch] is not None
        assert self._states[sub_batch] == 'step'
        self._states[sub_batch] = None
        obses, rews, dones, infos = ([], [], [], [])
        for env, action in zip(self.envs[sub_batch], self._step_actions[sub_batch]):
            obs, rew, done, info = env.step(action)
            if done:
                obs = env.reset()
            obses.append(obs)
            rews.append(rew)
            dones.append(done)
            infos.append(info)
        self._step_actions[sub_batch] = None
        return obses, rews, dones, infos

    def close(self):
        for batch in self.envs:
            for env in batch:
                env.close()


class BatchedWrapper(BatchedEnv):
    def __init__(self, env):
        self.env = env
        if hasattr(env, 'observation_space'):
            self.observation_space = env.observation_space
        if hasattr(env, 'action_space'):
            self.action_space = env.action_space

    @property
    def num_sub_batches(self):
        return self.env.num_sub_batches

    @property
    def num_envs_per_sub_batches(self):
        return self.env.num_envs_per_sub_batches

    def reset_start(self, sub_batch=0):
        self.env.reset_start(sub_batch=sub_batch)

    def reset_wait(self, sub_batch=0):
        return self.env.reset_wait(sub_batch=sub_batch)

    def step_start(self, actions, sub_batch=0):
        self.env.step_start(actions, sub_batch)

    def step_wait(self, sub_batch=0):
        return self.env.step_wait(sub_batch)

    def close(self):
        self.env.close()

class BatchedFrameStack(BatchedWrapper):
    def __init__(self, env, num_images=2, concat=True):
        super(BatchedFrameStack, self).__init__(env)
        self.concat = concat
        if hasattr(self, 'observation_space'):
            old = self.observation_space
            if concat:
                self.observation_space = gym.spaces.Box(np.repeat(old.low, num_images, axis=-1),
                                                        np.repeat(old.high, num_images, axis=-1),
                                                        dtype=old.dtype)
            else:
                self.observation_space = StackedBoxSpace(old, num_images)
        self._num_images = num_images
        self._history = [None] * env.num_sub_batches

    def reset_wait(self, sub_batch=0):
        obses = super(BatchedFrameStack, self).reset_wait(sub_batch)
        self._history[sub_batch] = [[o] * self._num_images for o  in obses]
        return self._packed_obs(sub_batch)

    def step_wait(self, sub_batch=0):
        obses, rews, dones, infos = super(BatchedFrameStack, self).step_wait(sub_batch)
        for i, (obs, done) in enumerate(zip(obses, dones)):
            if done:
                self._history[sub_batch][i] = [obs] * self._num_images
            else:
                self._history[sub_batch][i].append(obs)
                self._history[sub_batch][i] = self._history[sub_batch][i][1:]
        return self._packed_obs(sub_batch), rews, dones, infos

    def _packed_obs(self, sub_batch):
        if self.concat:
            return [np.concatenate(o, axis=-1) for o in self._history[sub_batch]]
        return [o.copy() for o in self._history[sub_batch]]
