"""
Policy classes updated for TensorFlow 2.x.
"""

import tensorflow as tf
import numpy as np
import gym
import logging
import copy
import tensorflow_probability as tfp

from .utils import RunningMeanStd  # Ensure this is compatible with TF 2.x

tfd = tfp.distributions


class Policy:
    def reset(self, **kwargs):
        pass

    def act(self, observation):
        raise NotImplementedError

    def get_variables(self):
        raise NotImplementedError


class MLPPolicy(Policy):
    def __init__(self, scope, *, ob_space, ac_space, hiddens,
                 normalize=False,
                 reuse=False):
        super().__init__()
        self.recurrent = False
        self.normalized = normalize

        self.scope = scope

        # Observation and action dimensions
        self.ob_dim = ob_space.shape[0]
        self.ac_dim = ac_space.shape[0]

        if self.normalized:
            if self.normalized != 'ob':
                self.ret_rms = RunningMeanStd(shape=())
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        # Build policy network
        self.policy_layers = []
        for i, hid_size in enumerate(hiddens):
            self.policy_layers.append(tf.keras.layers.Dense(
                hid_size, activation='tanh', name=f'{self.scope}_polfc{i+1}'))
        self.policy_mean = tf.keras.layers.Dense(
            self.ac_dim, name=f'{self.scope}_polfinal')

        # Build value network
        self.value_layers = []
        for i, hid_size in enumerate(hiddens):
            self.value_layers.append(tf.keras.layers.Dense(
                hid_size, activation='tanh', name=f'{self.scope}_vffc{i+1}'))
        self.value_output = tf.keras.layers.Dense(1, name=f'{self.scope}_vffinal')

        # Log std variable
        self.logstd = tf.Variable(
            initial_value=tf.zeros([1, self.ac_dim]), name=f'{self.scope}_logstd')

    def call(self, observations, stochastic=True):
        # observations is a batch of observations
        obz = observations
        if self.normalized:
            obz = tf.clip_by_value(
                (observations - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

        # Compute value
        v = obz
        for layer in self.value_layers:
            v = layer(v)
        vpredz = tf.squeeze(self.value_output(v), axis=-1)
        if self.normalized and self.normalized != 'ob':
            vpred = vpredz * self.ret_rms.std + self.ret_rms.mean
        else:
            vpred = vpredz

        # Compute policy mean
        p = obz
        for layer in self.policy_layers:
            p = layer(p)
        mean = self.policy_mean(p)
        logstd = self.logstd

        # Create a distribution
        dist = tfd.MultivariateNormalDiag(
            loc=mean, scale_diag=tf.exp(logstd))
        if stochastic:
            action = dist.sample()
        else:
            action = dist.mean()
        return action, vpred

    def act(self, observation, stochastic=True):
        observation = np.array(observation)
        observation = observation.reshape(1, -1)  # Ensure batch dimension
        action, vpred = self.call(observation, stochastic=stochastic)
        # Since we are in eager mode, action and vpred are tensors
        action = action.numpy()[0]
        vpred = vpred.numpy()[0]
        return action, {'vpred': vpred}

    def reset(self, **kwargs):
        pass

    def get_variables(self):
        return [var for var in self.__dict__.values() if isinstance(var, tf.Variable)] + \
               [var for layer in self.policy_layers + self.value_layers for var in layer.trainable_variables] + \
               self.policy_mean.trainable_variables + \
               self.value_output.trainable_variables


class LSTMPolicy(Policy):
    def __init__(self, scope, *, ob_space, ac_space, hiddens,
                 reuse=False, normalize=False):
        super().__init__()
        self.recurrent = True
        self.normalized = normalize

        self.scope = scope

        # Observation and action dimensions
        self.ob_dim = ob_space.shape[0]
        self.ac_dim = ac_space.shape[0]

        if self.normalized:
            if self.normalized != 'ob':
                self.ret_rms = RunningMeanStd(shape=())
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        # Build embedding layers (before LSTM)
        self.embedding_layers = []
        for i, hid_size in enumerate(hiddens[:-1]):
            self.embedding_layers.append(
                tf.keras.layers.Dense(hid_size, activation='tanh', name=f'{self.scope}_emb_fc{i+1}'))

        lstm_hidden_size = hiddens[-1]

        # Value LSTM cell
        self.value_lstm_cell = tf.keras.layers.LSTMCell(
            lstm_hidden_size, name=f'{self.scope}_lstmv')

        # Value output layer
        self.value_output = tf.keras.layers.Dense(1, name=f'{self.scope}_vffinal')

        # Policy LSTM cell
        self.policy_lstm_cell = tf.keras.layers.LSTMCell(
            lstm_hidden_size, name=f'{self.scope}_lstmp')

        # Policy mean output layer
        self.policy_mean = tf.keras.layers.Dense(self.ac_dim, name=f'{self.scope}_polfinal')

        # Log std variable
        self.logstd = tf.Variable(
            initial_value=tf.zeros([1, self.ac_dim]), name=f'{self.scope}_logstd')

        # Initialize states
        self.value_state = None
        self.policy_state = None

        self.reset()

    def act(self, observation, stochastic=True):
        observation = np.array(observation).reshape(1, -1)
        obz = observation.astype(np.float32)
        if self.normalized:
            obz = tf.clip_by_value(
                (obz - self.ob_rms.mean.numpy()) / self.ob_rms.std.numpy(), -5.0, 5.0)

        # Embedding
        x = obz
        for layer in self.embedding_layers:
            x = layer(x)

        # Value
        v_output, self.value_state = self.value_lstm_cell(
            x, states=self.value_state)
        vpredz = tf.squeeze(self.value_output(v_output), axis=-1)
        if self.normalized and self.normalized != 'ob':
            vpred = vpredz * self.ret_rms.std + self.ret_rms.mean
        else:
            vpred = vpredz

        # Policy
        p_output, self.policy_state = self.policy_lstm_cell(
            x, states=self.policy_state)
        mean = self.policy_mean(p_output)
        logstd = self.logstd

        dist = tfd.MultivariateNormalDiag(
            loc=mean, scale_diag=tf.exp(logstd))
        if stochastic:
            action = dist.sample()
        else:
            action = dist.mean()

        action = action.numpy()[0]
        vpred = vpred.numpy()[0]

        return action, {'vpred': vpred}

    def reset(self):
        self.value_state = self.value_lstm_cell.get_initial_state(batch_size=1, dtype=tf.float32)
        self.policy_state = self.policy_lstm_cell.get_initial_state(batch_size=1, dtype=tf.float32)

    def get_variables(self):
        return [var for var in self.__dict__.values() if isinstance(var, tf.Variable)] + \
               [var for layer in self.embedding_layers for var in layer.trainable_variables] + \
               self.value_lstm_cell.trainable_variables + \
               self.policy_lstm_cell.trainable_variables + \
               self.value_output.trainable_variables + \
               self.policy_mean.trainable_variables

