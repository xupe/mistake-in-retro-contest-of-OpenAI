'''
Prioritized Experience Replay( based on https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py )
'''
import numpy as py
import math

class SumTree(object):
     #data_pointer = 0

     def __init__(self, capacity):
         self.capacity = capacity
         self.tree = np.zeros(2 * capacity - 1)
         self.data = np.zeros(capacity, dtype=object)
         self.data_pointer = 0
         self.count = 0

     def add(self, p, data):
         tree_idx = self.data_pointer + self.capacity - 1
         data.update_weight(p)
         self.data[self.data_pointer] = data
         self.update(tree_idx, p)

         self.data_pointer += 1
         if self.count < self.capacity:
             self.count += 1
         if self.data_pointer >= self.capacity:
             self.data_pointer = 0

     def update(self, tree_idx, p):
         change = p - self.tree[tree_idx]
         self.tree[tree_idx] = p
         self.data[tree_idx - self.capacity + 1].update_weight(p)

         while tree_idx != 0:
             tree_idx = (tree_idx - 1) // 2
             self.tree[tree_idx] += change

     def get_leaf(self, v):
         parent_index = 0
         while True:
             l_index = 2 * parent_index + 1
             r_index = l_index + 1
             if l_index >= len(self.tree):
                 leaf_index = parent_index
                 break
             else:
                 if v <= self.tree[l_index]:
                     parent_index = l_index
                 else:
                     v -= self.tree[l_index]
                     parent_index = r_index

         data_index = leaf_index - self.capacity + 1
         return leaf_index, self.tree[leaf_index], self.data[data_index]

     def total_p(self):
         return self.tree[0]


class Memory(object):

     epsilon = 0.01  #avoid divide by zero
     alpha = 0.6     #TD_error importance
     beta = 0.4      #import sampling，
     beta_increment_per_sample = 0.001   #increment of beta per sample
     abs_err_upper = 1.    #max value of TD_error

     def __init__(self, capacity):
         self.tree = SumTree(capacity)

     def store(self, transition):
         max_p = np.max(self.tree.tree[-self.tree.capacity:])
         if max_p == 0:
             max_p = self.abs_err_upper
         self.tree.add(max_p, transition)
         #transition.update_weight(max_p)

     def sample(self, n):
         b_index, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n,),                dtype=object), np.empty((n, 1))
         pri_seg = self.tree.total_p() / n
         self.beta = np.min([1., self.beta + self.beta_increment_per_sample])

         if self.tree.count < self.tree.capacity:
             min_prob = np.min(self.tree.tree[-self.tree.capacity:-self.tree.capacity+self.tree.      count]) / self.tree.total_p()
         else:
             min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p()
         for i in range(n):
             a, b = pri_seg * i, pri_seg * (i + 1)
             v = np.random.uniform(a, b)
             index, p, data = self.tree.get_leaf(v)
             prob = p / self.tree.total_p()
             ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
             b_index[i], b_memory[i] = index, data
         return b_index, b_memory, ISWeights

     def batch_update(self, tree_index, abs_errors):
         abs_errors += self.epsilon
         clipping_errors = np.minimum(abs_errors, self.abs_err_upper)
         ps = np.power(clipping_errors, self.alpha)
         for ti, p in zip(tree_index, ps):
             self.tree.update(ti, p)
