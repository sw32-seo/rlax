"""An Episodic Actor-Critic agent trained to play BSuite's Catch env."""

import collections

import dm_env
import experiment
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import app, flags
from bsuite.environments import catch
from haiku import nets

import rlax

ActorOutput = collections.namedtuple("ActorOutput", ["actions", "logits", "q_values"])

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("train_episodes", 10000, "Number of train episodes.")
flags.DEFINE_integer("num_hidden_units", 50, "Number of network hidden units.")
flags.DEFINE_integer("sequence_length", 500, "Length of (action, timestep) sequences.")
flags.DEFINE_float("epsilon", 0.01, "Epsilon-greedy exploration probability.")
flags.DEFINE_float("lambda_", 0.9, "Mixing parameter for Q(lambda).")
flags.DEFINE_float("discount_factor", 0.99, "Q-learning discount factor.")
flags.DEFINE_float("learning_rate", 0.001, "Optimizer learning rate.")
flags.DEFINE_integer("eval_episodes", 100, "Number of evaluation episodes.")
flags.DEFINE_integer("evaluate_every", 50, "Number of episodes between evaluations.")


def build_network(num_hidden_units: int, num_actions: int) -> hk.Transformed:
    """Factory for a simple MLP network for approximating Q-values."""

    def pi(obs):
        flatten = lambda x: jnp.reshape(x, (-1,))
        network = hk.Sequential([flatten, nets.MLP([num_hidden_units, num_actions + 1])])
        return network(obs)

    return hk.without_apply_rng(hk.transform(pi))


class SequenceWithLogitsAccumulator:
    """Accumulator for gathering the latest timesteps into sequences including logit_t.

    Note that accumulator ready when the episode is over.
  """

    def __init__(self, length):
        self._timesteps = collections.deque(maxlen=length)

    def push(self, timestep, action):
        # Replace `None`s with zeros as these will be put into NumPy arrays.
        a_tm1 = 0 if action is None else action
        timestep_t = timestep._replace(
            step_type=int(timestep.step_type),
            reward=0. if timestep.reward is None else timestep.reward,
            discount=0. if timestep.discount is None else timestep.discount,
        )
        self._timesteps.append((a_tm1, timestep_t))

    def sample(self, batch_size):
        """Returns current sequence of accumulated timesteps."""
        if batch_size != 1:
            raise ValueError("Require batch_size == 1.")

        actions, timesteps = jax.tree_map(lambda *ts: np.stack(ts), *self._timesteps)
        self._timesteps.clear()
        return actions, timesteps

    def is_ready(self, batch_size):
        if batch_size != 1:
            raise ValueError("Require batch_size == 1.")
        return self._timesteps[-1][-1].last()


class EpisodicActorCritic:
    """An Episodic Actor-Critic algorithm."""

    def __init__(self, observation_spec, action_spec, num_hidden_units, epsilon, lambda_, learning_rate):
        self._observation_spec = observation_spec
        self._action_spec = action_spec
        self._epsilon = epsilon
        self._lambda = lambda_

        # Neural net and optimiser.
        self._network = build_network(num_hidden_units, action_spec.num_values)

        self._optimizer = optax.adam(learning_rate)

    def initial_params(self, key):
        sample_input = self._observation_spec.generate_value()
        return self._network.init(key, sample_input)

    def initial_actor_state(self):
        return ()

    def initial_learner_state(self, params):
        return self._optimizer.init(params)

    # jitting for speed
    @jax.jit
    def actor_step(self, params, env_output, actor_state, key, evaluation):
        output = self._network.apply(params, env_output.observation)
        policy_logit = output[:-1]
        value = output[-1]

        a = hk.multinomial(key, policy_logit, num_samples=1)
        return ActorOutput(actions=a, logits=policy_logit, q_values=value), actor_state

    # jitting for speed
    @jax.jit
    def learner_step(self, params, data, learner_state, unused_key):
        dloss_dtheta = jax.grad(self._loss)(params, *data)
        updates, learner_state = self._optimizer.update(dloss_dtheta, learner_state)
        params = optax.apply_updates(params, updates)
        return params, learner_state

    def _loss(self, params, actions, timesteps):
        """Calculates REINFORCE loss given parameters, actions and timesteps, and """
        network_apply_sequence = jax.vmap(self._network.apply, in_axes=(None, 0))
        outputs = network_apply_sequence(params, timesteps.observation)
        policy_logits = outputs[:, :-1]
        values = outputs[:, -1]

        # Use a mask since the sequence could cross episode boundaries.
        mask = jnp.not_equal(timesteps.step_type, int(dm_env.StepType.LAST))

        # Discount ought to be zero on a LAST timestep, use the mask to ensure this.
        discount_t = timesteps.discount[1:] * mask[1:]

        td_errors = rlax.td_lambda(
            v_tm1=values[:-1],
            r_t=timesteps.reward[1:],
            discount_t=discount_t,
            v_t=values[1:],
            lambda_=jnp.array(self._lambda),
        )
        critic_loss = jnp.mean(td_errors**2)

        # Mask out TD errors for the last state in an episode.
        policy_loss = rlax.policy_gradient_loss(policy_logits[:-1], actions[1:], adv_t=td_errors, w_t=jnp.ones((1,)))
        return policy_loss + critic_loss


def main(unused_arg):
    env = catch.Catch(seed=FLAGS.seed)
    agent = EpisodicActorCritic(observation_spec=env.observation_spec(),
                                action_spec=env.action_spec(),
                                num_hidden_units=FLAGS.num_hidden_units,
                                epsilon=FLAGS.epsilon,
                                lambda_=FLAGS.lambda_,
                                learning_rate=FLAGS.learning_rate)

    accumulator = SequenceWithLogitsAccumulator(length=FLAGS.sequence_length)
    experiment.run_loop(
        agent=agent,
        environment=env,
        accumulator=accumulator,
        seed=FLAGS.seed,
        batch_size=1,
        train_episodes=FLAGS.train_episodes,
        evaluate_every=FLAGS.evaluate_every,
        eval_episodes=FLAGS.eval_episodes,
    )


if __name__ == "__main__":
    app.run(main)
