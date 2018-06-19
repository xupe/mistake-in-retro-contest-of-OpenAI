from abc import ABC, abstractmethod, abstractproperty
from math import sqrt

import random
import numpy as np

class ReplayBuffer(ABC):
    @abstractproperty
    def size(self):
        pass

    @abstractmethod
    def sample(self, num_samples):
        pass

    @abstractmethod
    def add_sample(self, sample, init_weight=None):
        pass

    @abstractmethod
    def update_weights(self, samples, new_weights):
        pass


class FloatBuffer:
    def __init__(self, capacity, dtype='float64'):
        self._capacity = capacity
        self._start = 0
        self._used = 0
        self._buffer = np.zeros((capacity, ), dtype=dtype)
        self._bin_size = int(sqrt(capacity))
        num_bins = capacity // self._bin_size
        if num_bins * self._bin_size < capacity:
            num_bins += 1
        self._bin_sums = np.zeros((num_bins,), dtype=dtype)
        self._min = 0


    def append(self, value):
        index = (self._start + self._used) % self._capacity
        if self._used < self._capacity:
            self._used += 1
        else:
            self._start = (self._start + 1) % self._capacity
        self._set_index(index, value)


    def _set_index(self, index, value):
        assert not np.isnan(value)
        assert value > 0
        needs_recompute = False
        if self._min == self._buffer[index]:
            needs_recompute = True
        elif value < self._min:
            self._min = value
        bin_index = index // self._bin_size
        old_value = self._buffer[index]
        self._buffer[index] = value
        self._bin_sums[bin_index] += (value - old_value)
        if needs_recompute:
            self._recompute_min()


    def _recompute_min(self):
        if self._used < self._capacity:
            self._min = np.min(self._buffer[:self._used])
        else:
            self._min = np.min(self._buffer)


    def _bin(self, bin_index):
        if bin_index == len(self._bin_sums) - 1:
            return self._buffer[self._bin_size*bin_index:]
        return self._buffer[self._bin_size*bin_index : self._bin_size*(bin_index+1)]


    def sample(self, num_values):
        assert self._used > num_values
        res = []
        probs = []
        bin_probs = self._bin_sums / np.sum(self._bin_sums)
        while len(res) < num_values:
            bin_index = np.random.choice(a=len(self._bin_sums), p=bin_probs)
            bin_values = self._bin(bin_index)
            sub_probs = bin_values / self._bin_sums[bin_index]
            sub_index = np.random.choice(a=len(sub_probs), p=sub_probs)
            index = bin_index * self._bin_size + sub_index
            res.append(index)
            probs.append(bin_probs[bin_index] * sub_probs[sub_index])
        return (np.array(list(res)) - self._start) % self._capacity, np.array(probs)


    def set_value(self, index, value):
        index = (index + self._start) % self._capacity
        self._set_index(index, value)


    def min(self):
        return self._min


    def sum(self):
        return np.sum(self._bin_sums)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, alpha, beta, first_max=1, epsilon=0):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.transitions = []
        self.errors = FloatBuffer(capacity)
        self._max_weight_arg = first_max


    def size(self):
        return len(self.transitions)


    def sample(self, num_samples):
        indices, probs = self.errors.sample(num_samples)
        beta = float(self.beta)
        importance_weights = np.power(probs * self.size(), -beta)
        importance_weights /= np.power(self.errors.min() / self.errors.sum() * self.size(), -beta)
        samples = []
        for i, weight in zip(indices, importance_weights):
            sample = self.transitions[i].copy()
            sample['weight'] = weight
            sample['id'] = i
            samples.append(sample)
        return samples


    def add_sample(self, sample, init_weight=None):
        self.transitions.append(sample)
        if init_weight is None:
            self.errors.append(self._process_weight(self._max_weight_arg))
        else:
            self.errors.append(self._process_weight(init_weight))
        while len(self.transitions) > self.capacity:
            del self.transitions[0]


    def update_weights(self, samples, new_weights):
        for sample, weight in zip(samples, new_weights):
            self.errors.set_value(sample['id'], self._process_weight(weight))


    def _process_weight(self, weight):
        self._max_weight_arg = max(self._max_weight_arg, weight)
        return (weight + self.epsilon) ** self.alpha



