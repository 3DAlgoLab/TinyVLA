# coding=utf-8
# Copyright 2020 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stationary Stochastic Python Bandit environment with structured features."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Callable, Optional, Sequence, Text

import gin
import numpy as np
import tensorflow as tf
from tf_agents.bandits.environments import bandit_py_environment
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.specs import array_spec
from tf_agents.typing import types
from tf_agents.utils import nest_utils


GLOBAL_KEY = bandit_spec_utils.GLOBAL_FEATURE_KEY
PER_ARM_KEY = bandit_spec_utils.PER_ARM_FEATURE_KEY


@gin.configurable
class StationaryStochasticStructuredPyEnvironment(
    bandit_py_environment.BanditPyEnvironment
):
  """Stationary Stochastic Bandit environment with structured features.

  This environment can generate global and per-arm observations of any nested
  structure.
  """

  def __init__(
      self,
      global_context_sampling_fn: Callable[[], types.Array],
      arm_context_sampling_fn: Callable[[], types.Array],
      num_actions: int,
      reward_fn: Callable[[types.Array], Sequence[float]],
      batch_size: Optional[int] = 1,
      name: Optional[Text] = 'stationary_stochastic_structured',
  ):
    """Initializes the environment.

    In each round, global context is generated by global_context_sampling_fn,
    per-arm contexts are generated by arm_context_sampling_fn.

    The two feature generating functions should output a single observation, not
    including either the batch_size or the number of actions.

    The reward_fn function takes a global and a per-arm feature, and outputs a
    possibly random reward.

    Example:
      def global_context_sampling_fn():
        return np.random.randint(0, 10, [2])  # 2-dimensional global features.

      def arm_context_sampling_fn():
        return {'armf1': np.random.randint(-3, 4, [3]),    # A dictionary of
                'armf2': np.random.randint(0, 2, [4, 5])}  # arm features.

      def reward_fn(global, arm):
        return sum(global) + arm['armf1'][0] + arm['armf2'][3, 3]

      env = StationaryStochasticPyEnvironment(global_context_sampling_fn,
                                              arm_context_sampling_fn,
                                              5,
                                              reward_fn,
                                              batch_size=5)

    Args:
      global_context_sampling_fn: A function that outputs a possibly nested
        structure of features. This output is the global context. Its shapes and
        types must be consistent accross calls.
      arm_context_sampling_fn: A function that outputs a possibly nested
        structure of features. This output is the per-arm context. Its shapes
        must be consistent accross calls.
      num_actions: (int) the number of actions in every sample.
      reward_fn: A function that generates a reward when called with a global
        and a per-arm observation.
      batch_size: The batch size.
      name: The name of this environment instance.
    """
    self._global_context_sampling_fn = global_context_sampling_fn
    self._arm_context_sampling_fn = arm_context_sampling_fn
    self._num_actions = num_actions
    self._reward_fn = reward_fn
    self._batch_size = batch_size

    global_example = global_context_sampling_fn()
    arm_example = arm_context_sampling_fn()
    observation_spec = {
        GLOBAL_KEY: tf.nest.map_structure(
            array_spec.ArraySpec.from_array, global_example
        ),
        PER_ARM_KEY: array_spec.add_outer_dims_nest(
            tf.nest.map_structure(array_spec.ArraySpec.from_array, arm_example),
            (num_actions,),
        ),
    }

    action_spec = array_spec.BoundedArraySpec(
        shape=(),
        dtype=np.int32,
        minimum=0,
        maximum=num_actions - 1,
        name='action',
    )

    super(StationaryStochasticStructuredPyEnvironment, self).__init__(
        observation_spec, action_spec, name=name
    )

  def batched(self) -> bool:
    return True

  @property
  def batch_size(self) -> Optional[int]:
    return self._batch_size

  def _generate_batch_of_observations(self, generator_fn, num_samples):
    unstacked_obs = [generator_fn() for _ in range(num_samples)]
    return nest_utils.stack_nested_arrays(unstacked_obs)

  def _observe(self) -> types.NestedArray:
    global_obs = self._generate_batch_of_observations(
        self._global_context_sampling_fn, self._batch_size
    )
    arm_obs = self._generate_batch_of_observations(
        self._arm_context_sampling_fn, self._batch_size * self._num_actions
    )
    arm_obs = tf.nest.map_structure(
        lambda x: x.reshape((self.batch_size, self._num_actions) + x.shape[1:]),
        arm_obs,
    )
    self._observation = {GLOBAL_KEY: global_obs, PER_ARM_KEY: arm_obs}
    return self._observation

  def _apply_action(self, action: types.Array) -> types.Array:
    if len(action) != self.batch_size:
      raise ValueError('Number of actions must match batch size.')
    global_obs = self._observation[GLOBAL_KEY]  # pytype: disable=attribute-error  # trace-all-classes
    batch_size_range = list(range(self.batch_size))
    arm_obs = tf.nest.map_structure(
        lambda x: x[batch_size_range, action, :], self._observation[PER_ARM_KEY]  # pytype: disable=attribute-error  # trace-all-classes
    )

    def _get_element_from_batch(structure, index):
      return tf.nest.map_structure(lambda x: x[index], structure)

    reward = np.stack(
        [
            # pytype: disable=wrong-arg-count  # trace-all-classes
            self._reward_fn(
                _get_element_from_batch(global_obs, b),
                _get_element_from_batch(arm_obs, b),
            )
            for b in batch_size_range
            # pytype: enable=wrong-arg-count
        ]
    )
    return reward