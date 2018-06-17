import numpy as np
import tensorflow as tf

def NStepTransition(s, a, r, ss, gamma):
    total_reward = 0
    pre = 1
    for reward in r:
        total_reward += pre * reward
        pre *= gamma
    return np.hstack(s, a, total_reward, ss)

def take_vector_elems(vectors, indices):
    return tf.gather_nd(vectors, tf.stack([tf.range(tf.shape(vectors)[0]), indices], axis=1))



