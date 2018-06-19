import tensorflow as tf
import sys
sys.path.append('..')

from rollouts import PrioritizedReplayBuffer, NStepPlayer, BatchedPlayer
from envs import BatchedFrameStack, BatchedGymEnv
from utils import AllowBacktracking, make_env
from models import DQN, rainbow_models
from spaces import gym_space_vectorizer

import gym_remote.exceptions as gre

def main():
    """Run DQN until the environment throws an exception."""
    env = AllowBacktracking(make_env(stack=False, scale_rew=False))
    env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    with tf.Session(config=config) as sess:
        dqn = DQN(*rainbow_models(sess,
                                  env.action_space.n,
                                  gym_space_vectorizer(env.observation_space),
                                  min_val=-200,
                                  max_val=200))
        player = NStepPlayer(BatchedPlayer(env, dqn.online_net), 4)
        saver = tf.train.Saver()
        saver.restore(sess, 'transfer_model/name of your pre-training model')
        optim, optimize = dqn.optimize(learning_rate=0.0001)
        sess.run(tf.variables_initializer(optim.variables()))
        #sess.run(tf.global_variables_initializer())
        dqn.train(num_steps=3000000, # Make sure an exception arrives before we stop.
                  player=player,
                  replay_buffer=PrioritizedReplayBuffer(500000, 0.5, 0.4, epsilon=0.1),
                  optimize_op=optimize,
                  train_interval=1,
                  target_interval=1024,
                  batch_size=32,
                  min_buffer_size=20000)

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
