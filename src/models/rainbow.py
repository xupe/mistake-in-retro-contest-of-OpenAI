# coding: utf8
import tensorflow as tf
import numpy as np
import math
from abc import abstractmethod
from functools import partial
import time
import sys
sys.path.append("..")

from rollouts import SumTree, Memory
from .base import TFQNetwork
from utils import take_vector_elems
from .BaseNet import nature_cnn, noisy_net_dense, sample_noise, nature_cnn_add_one_layer, my_net

class DistQNetwork(TFQNetwork):
    def __init__(self, session, num_actions, obs_vectorizer, name, num_atoms, min_val, max_val,
                 dueling=False, dense=tf.layers.dense):
        super(DistQNetwork, self).__init__(session, num_actions, obs_vectorizer, name)
        self.dueling = dueling
        self.dense = dense
        self.dist = ActionDist(num_atoms, min_val, max_val)
        old_vars = tf.trainable_variables()

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self.step_obs_ph = tf.placeholder(self.input_dtype, shape=(None, ) + obs_vectorizer.out_shape)
            self.step_base_out = self.base(self.step_obs_ph) #self.weight_noisy0_0, self.bias_noisy0_0)
            log_probs = self.value_func(self.step_base_out)
            values = self.dist.mean(log_probs)
            self.step_outs = (values, log_probs)
        self.variables = [v for v in tf.trainable_variables() if v not in old_vars]

    @property
    def stateful(self):
        return False

    @abstractmethod
    def base(self, obs_batch):
        pass

    def start_state(self, batch_size):
        return None

    def step(self, observations, states):
        feed = self.step_feed_dict(observations, states)
        values, dists = self.session.run(self.step_outs, feed_dict=feed)
        return {
            'actions': np.argmax(values, axis=1),
            'states': None,
            'action_values': values,
            'action_dists': dists
        }

    def transition_loss(self, target_net, obses, actions, rews, new_obses, terminals, discounts):
        with tf.variable_scope(target_net.name, reuse=True):
            max_actions = tf.argmax(target_net.dist.mean(target_net.value_func(target_net.base(new_obses))),
                                    axis=1, output_type=tf.int32)
        with tf.variable_scope(target_net.name, reuse=True):
            target_preds = target_net.value_func(target_net.base(new_obses))
            target_preds = tf.where(terminals,
                                    tf.zeros_like(target_preds) - math.log(self.dist.num_atoms),
                                    target_preds)
        discounts = tf.where(terminals, tf.zeros_like(discounts), discounts)
        target_dists = self.dist.add_rewards(tf.exp(take_vector_elems(target_preds, max_actions)), rews, discounts)
        with tf.variable_scope(self.name, reuse=True):
            online_preds = self.value_func(self.base(obses))
            onlines = take_vector_elems(online_preds, actions)
            return _kl_divergence(tf.stop_gradient(target_dists), onlines)

    @property
    def input_dtype(self):
        return tf.float32

    def value_func(self, feature_batch):
        input_nums = feature_batch.get_shape()[-1].value
        logits = self.dense(feature_batch, self.num_actions * self.dist.num_atoms)
        actions = tf.reshape(logits, (tf.shape(logits)[0], self.num_actions, self.dist.num_atoms))
        if not self.dueling:
            return tf.nn.log_softmax(actions)
        values = tf.expand_dims(self.dense(feature_batch, self.dist.num_atoms), axis=1)
        actions -= tf.reduce_mean(actions, axis=1, keep_dims=True)
        return tf.nn.log_softmax(values + actions)


    def step_feed_dict(self, observations, states):
        return {self.step_obs_ph: self.obs_vectorizer.to_vecs(observations),
                }

def _kl_divergence(probs, log_probs):
    masked_diff = tf.where(tf.equal(probs, 0), tf.zeros_like(probs), tf.log(probs) - log_probs)
    return tf.reduce_sum(probs * masked_diff, axis=-1)

class NatureDistQNetwork(DistQNetwork):
    def __init__(self,
                 session,
                 num_actions,
                 obs_vectorizer,
                 name,
                 num_atoms,
                 min_val,
                 max_val,
                 dueling=False,
                 dense=tf.layers.dense,
                 input_dtype=tf.uint8,
                 #归一化
                 input_scale=1 / 0xff):
        self._input_dtype = input_dtype
        self.input_scale = input_scale
        super(NatureDistQNetwork, self).__init__(session, num_actions, obs_vectorizer,
                                                 name, num_atoms, min_val, max_val,
                                                 dueling=dueling, dense=dense)

    @property
    def input_dtype(self):
        return self._input_dtype

    def base(self, obs_batch):
        obs_batch = tf.cast(obs_batch, tf.float32) * self.input_scale
        #return my_net(obs_batch, dense=self.dense)
        return nature_cnn_add_one_layer(obs_batch, dense=self.dense)
        #return nature_cnn(obs_batch, dense=self.dense)

class ActionDist:
    def __init__(self, num_atoms, min_val, max_val):
        assert num_atoms >= 2
        assert max_val >min_val
        self.num_atoms = num_atoms
        self.min_val = min_val
        self.max_val = max_val
        self._delta = (self.max_val - self.min_val) / (self.num_atoms - 1)

    def atom_values(self):
        return [self.min_val + i * self._delta for i in range(self.num_atoms)]

    def mean(self, log_probs):
        probs = tf.exp(log_probs)
        return tf.reduce_sum(probs * tf.constant(self.atom_values(), dtype=probs.dtype), axis=-1)

    def add_rewards(self, probs, rewards, discounts):
        #probs: N * atoms
        atom_rews = tf.tile(tf.constant([self.atom_values()], dtype=probs.dtype),
                            tf.stack([tf.shape(rewards)[0], 1]))
        fuzzy_index = tf.expand_dims(rewards, axis=1) + tf.expand_dims(discounts, axis=1) * atom_rews
        fuzzy_index = (fuzzy_index - self.min_val) / self._delta

        fuzzy_index = tf.clip_by_value(fuzzy_index, 1e-18, float(self.num_atoms - 1))

        indices_1 = tf.cast(tf.ceil(fuzzy_index) - 1, tf.int32)
        fracs_1 = tf.abs(tf.ceil(fuzzy_index) - fuzzy_index)
        indices_2 = indices_1 + 1
        fracs_2 = 1 - fracs_1

        res = tf.zeros_like(probs)
        for indices, fracs in [(indices_1, fracs_1), (indices_2, fracs_2)]:
            index_matrix = tf.expand_dims(tf.range(tf.shape(indices)[0], dtype=tf.int32), axis=1)
            index_matrix = tf.tile(index_matrix, (1, self.num_atoms))
            scatter_indices = tf.stack([index_matrix, indices], axis=-1)
            res = res + tf.scatter_nd(scatter_indices, probs * fracs, tf.shape(res))
        return res

def rainbow_models(session,
                   num_actions, obs_vectorizer,
                   num_atoms=51,
                   min_val=-10,
                   max_val=10,
                   sigma0=0.5):
    maker = lambda name: NatureDistQNetwork(session, num_actions, obs_vectorizer, name,
                                            num_atoms, min_val, max_val, dueling=True,
                                            dense=partial(noisy_net_dense, sigma0=sigma0))
    return maker('online'), maker('target')

class DQN:
    def __init__(self, online_net, target_net, discount=0.99):
        """
        Create a Q-learning session.

        Args:
          online_net: the online TFQNetwork.
          target_net: the target TFQNetwork.
          discount: the per-step discount factor.
        """
        self.online_net = online_net
        self.target_net = target_net
        self.discount = discount
        self.saver = tf.train.Saver()

        obs_shape = (None,) + online_net.obs_vectorizer.out_shape
        self.obses_ph = tf.placeholder(online_net.input_dtype, shape=obs_shape)
        self.actions_ph = tf.placeholder(tf.int32, shape=(None,))
        self.rews_ph = tf.placeholder(tf.float32, shape=(None,))
        self.new_obses_ph = tf.placeholder(online_net.input_dtype, shape=obs_shape)
        self.terminals_ph = tf.placeholder(tf.bool, shape=(None,))
        self.discounts_ph = tf.placeholder(tf.float32, shape=(None,))
        self.weights_ph = tf.placeholder(tf.float32, shape=(None,))

        losses = online_net.transition_loss(target_net, self.obses_ph, self.actions_ph,
                                            self.rews_ph, self.new_obses_ph, self.terminals_ph,
                                            self.discounts_ph)
        self.losses = self.weights_ph * losses
        self.loss = tf.reduce_mean(self.losses)

        assigns = []
        for dst, src in zip(target_net.variables, online_net.variables):
            assigns.append(tf.assign(dst, src))
        self.update_target = tf.group(*assigns)

    def feed_dict(self, transitions):
        """
        Generate a feed_dict that feeds the batch of
        transitions to the DQN loss terms.

        Args:
          transition: a sequence of transition dicts, as
            defined in anyrl.rollouts.ReplayBuffer.

        Returns:
          A dict which can be fed to tf.Session.run().
        """
        obs_vect = self.online_net.obs_vectorizer
        res = {
            self.obses_ph: obs_vect.to_vecs([t['obs'] for t in transitions]),
            self.actions_ph: [t['model_outs']['actions'][0] for t in transitions],
            self.rews_ph: [self._discounted_rewards(t['rewards']) for t in transitions],
            self.terminals_ph: [t['new_obs'] is None for t in transitions],
            self.discounts_ph: [(self.discount ** len(t['rewards'])) for t in transitions],
            self.weights_ph: [t['weight'] for t in transitions]
        }
        new_obses = []
        for trans in transitions:
            if trans['new_obs'] is None:
                new_obses.append(trans['obs'])
            else:
                new_obses.append(trans['new_obs'])
        res[self.new_obses_ph] = obs_vect.to_vecs(new_obses)
        return res

    def optimize(self, learning_rate=6.25e-5, epsilon=1.5e-4, **adam_kwargs):
        """
        Create a TF Op that optimizes the objective.

        Args:
          learning_rate: the Adam learning rate.
          epsilon: the Adam epsilon.
        """
        optim = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon, **adam_kwargs)
        return optim, optim.minimize(self.loss)

    def train(self,
              num_steps,
              player,
              replay_buffer,
              optimize_op,
              train_interval=1,
              target_interval=8192,
              batch_size=32,
              min_buffer_size=20000,
              tf_schedules=(),
              handle_ep=lambda steps, rew: None,
              timeout=None,
              save_iters=1024):
        """
        Run an automated training loop.

        This is meant to provide a convenient way to run a
        standard training loop without any modifications.
        You may get more flexibility by writing your own
        training loop.

        Args:
          num_steps: the number of timesteps to run.
          player: the Player for gathering experience.
          replay_buffer: the ReplayBuffer for experience.
          optimize_op: a TF Op to optimize the model.
          train_interval: timesteps per training step.
          target_interval: number of timesteps between
            target network updates.
          batch_size: the size of experience mini-batches.
          min_buffer_size: minimum replay buffer size
            before training is performed.
          tf_schedules: a sequence of TFSchedules that are
            updated with the number of steps taken.
          handle_ep: called with information about every
            completed episode.
          timeout: if set, this is a number of seconds
            after which the training loop should exit.
        """
        sess = self.online_net.session
        sess.run(self.update_target)
        steps_taken = 0
        next_target_update = target_interval
        next_train_step = train_interval
        start_time = time.time()
        while steps_taken < num_steps:
            if timeout is not None and time.time() - start_time > timeout:
                return
            transitions = player.play()
            for trans in transitions:
                if trans['is_last']:
                    handle_ep(trans['episode_step'] + 1, trans['total_reward'])
                replay_buffer.add_sample(trans)
                steps_taken += 1
                for sched in tf_schedules:
                    sched.add_time(sess, 1)
                if replay_buffer.size() >= min_buffer_size and steps_taken >= next_train_step:
                    next_train_step = steps_taken + train_interval
                    batch = replay_buffer.sample(batch_size)
                    _, losses = sess.run((optimize_op, self.losses),
                                         feed_dict=self.feed_dict(batch))
                    replay_buffer.update_weights(batch, losses)
                    '''loss = tf.reduce_mean(losses)
                    loss_value = loss.eval()
                    print('learn steps ' + str(steps_taken) + ' : loss is: ' + str(loss_value))
                    del loss_value
                    if steps_taken % save_iters == 0:
                        print('save model')
                        self.saver.save(sess=sess, save_path='./model/Sonic', global_step=steps_taken)'''
                if steps_taken >= next_target_update:
                    next_target_update = steps_taken + target_interval
                    sess.run(self.update_target)

    def _discounted_rewards(self, rews):
        res = 0
        for i, rew in enumerate(rews):
            res += rew * (self.discount ** i)
        return res


class RainBow:
    '''
    At first I use this to train the model of rainbow,
    and at last I choose the OpenAI baselines's way to train.
    most of them is same, except replay memory and player
    '''
    def __init__(self, session, num_actions, obs_vectorizer, num_atoms, min_val, max_val, sigma0, memory_size,
                 replace_iters, learning_rate, batch_size, discount, input_shape):
        self.session = session
        maker = lambda name: NatureDistQNetwork(self.session, num_actions, obs_vectorizer, name,
                                                num_atoms, min_val, max_val, dueling=True,
                                                dense=partial(noisy_net_dense, sigma0=sigma0))
        self.online = maker('online')
        print('online create over')
        self.target = maker('target')
        print('target create over')
        self.Memory = Memory(memory_size)
        self.replace_iters = replace_iters
        self.learn_step = 0
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.discount = discount
        self.input_shape = input_shape
        self.trans_size = 0

        obs_shape = (None, ) + self.input_shape
        self.obses_ph = tf.placeholder(self.online.input_dtype, shape=obs_shape)
        self.actions_ph = tf.placeholder(tf.int32, shape=(None,))
        self.rewards_ph = tf.placeholder(tf.float32, shape=(None,))
        self.new_obses_ph = tf.placeholder(self.online.input_dtype, shape=obs_shape)
        self.terminals_ph = tf.placeholder(tf.bool, shape=(None,))
        self.discounts_ph = tf.placeholder(tf.float32, shape=(None,))
        self.weights_ph = tf.placeholder(tf.float32, shape=(None,))

        self.losses = self.online.transition_loss(self.target, self.obses_ph, self.actions_ph,
                                             self.rewards_ph, self.new_obses_ph, self.terminals_ph,
                                             self.discounts_ph)
        self.losses = self.weights_ph * self.losses
        self.loss = tf.reduce_mean(self.losses)
        self.opti = self.optimize()

        t_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')
        o_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='online')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, o) for t, o in zip(t_param, o_param)]

        #self.session.run(tf.global_variables_initializer())

    def store_transition(self, transition):
        if not hasattr(self, 'memory_count'):
            self.memory_count = 0
        self.trans_size += 1
        #index = self.memory_count % self.memory_size

        self.Memory.store(transition)

    def choose_action(self, observation, states):
        observation = observation[np.newaxis, :]
        output = self.online.step(observation, states)
        action = output['actions'][0]
        return action, output

    def feed_dict(self, transitions):
        res = {
            self.obses_ph: [t.transfer()['obs'] for t in transitions],
            self.actions_ph: [t.transfer()['model_outs']['actions'][0] for t in transitions],
            self.rewards_ph: [self._discount_rewards(t.transfer()['rewards']) for t in transitions],
            self.terminals_ph: [t.transfer()['new_obs'] is None for t in transitions],
            self.discounts_ph: [(self.discount ** len(t.transfer()['rewards'])) for t in transitions],
            self.weights_ph: [t.transfer()['weight'] for t in transitions]
        }
        new_obs = []
        for trans in transitions:
            if trans.transfer()['new_obs'] is None:
                new_obs.append(trans.transfer()['obs'])
            else:
                new_obs.append(trans.transfer()['new_obs'])
        res[self.new_obses_ph] = new_obs
        return res

    def optimize(self, epsilon=1.5e-4, **adam_kwargs):
        optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=epsilon, **adam_kwargs)
        return optim.minimize(self.loss)

    def _discount_rewards(self, rewards):
        res = 0
        for i, rew in enumerate(rewards):
            res += rew * (self.discount ** i)
        return res

    def learn(self):
        print('learn')
        if self.learn_step % self.replace_iters == 0:
            self.session.run(self.target_replace_op)
            print('target params replace')

        tree_index, batch_memory, ISWeights = self.Memory.sample(self.batch_size)
        _, losses = self.session.run((self.opti, self.losses),
                                     feed_dict=self.feed_dict(batch_memory))
        self.Memory.batch_update(tree_index, losses)

        self.learn_step += 1
        return tf.reduce_mean(losses)


class BoxGaussian():
    """
    A probability distribution over continuous variables,
    parameterized as a diagonal gaussian.
    """
    def __init__(self, low, high):
        self.low = low
        self.high = high

    @property
    def out_shape(self):
        return self.low.shape

    def to_vecs(self, space_elements):
        return np.array(space_elements)

class Transition(object):
    def __init__(self, obs, model_outs, rewards, new_obs, weight=1):
        self.obs = obs
        self.model_outs = model_outs
        self.rewards = rewards
        self.new_obs = new_obs
        self.weight = weight

    def transfer(self):
        res = {
            'obs': self.obs,
            'model_outs': self.model_outs,
            'rewards': self.rewards,
            'new_obs': self.new_obs,
            'weight': self.weight
        }
        return res

    def update_weight(self, weight):
        self.weight = weight

    @property
    def size(self):
        return

def print_test():
    print('test')


