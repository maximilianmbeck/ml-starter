from dataclasses import dataclass
from typing import cast

import gymnasium as gym
import numpy as np
import scipy
import torch
from torch import Tensor
from torch.distributions.normal import Normal
from torch.nn import functional as F

from ml.core.config import conf_field
from ml.core.registry import register_task
from ml.core.state import State
from ml.models.a2c import SimpleA2CModel
from ml.tasks.environments.base import Environment
from ml.tasks.rl.base import ReinforcementLearningTask, ReinforcementLearningTaskConfig
from ml.utils.logging import configure_logging


@dataclass
class BWAction:
    """Defines the action space for the BipedalWalker task."""

    hip_1: float | Tensor
    knee_1: float | Tensor
    hip_2: float | Tensor
    knee_2: float | Tensor
    log_prob: list[float] | list[Tensor]
    value: float | Tensor | None = None
    advantage: float | Tensor | None = None
    returns: float | Tensor | None = None

    @classmethod
    def from_policy(cls, policy: Tensor, log_prob: Tensor, value: Tensor) -> "BWAction":
        assert policy.shape == (4,) and log_prob.shape == (4,) and value.shape == (1,)
        log_prob_list = log_prob.detach().cpu().tolist()
        return cls(policy[0].item(), policy[1].item(), policy[2].item(), policy[3].item(), log_prob_list, value.item())

    def to_tensor(self) -> Tensor:
        assert isinstance(self.hip_1, Tensor) and isinstance(self.knee_1, Tensor)
        assert isinstance(self.hip_2, Tensor) and isinstance(self.knee_2, Tensor)
        return torch.stack([self.hip_1, self.knee_1, self.hip_2, self.knee_2], dim=-1)


@dataclass
class BWState:
    """Defines the state space for the BipedalWalker task."""

    observation: Tensor
    reward: float | Tensor
    terminated: bool | Tensor
    truncated: bool | Tensor
    info: dict | Tensor
    reset: bool | Tensor


class BipedalWalkerEnvironment(Environment[BWState, BWAction]):
    def __init__(self, hardcore: bool = False) -> None:
        super().__init__()

        self.env = gym.make("BipedalWalker-v3", hardcore=hardcore, render_mode="rgb_array")

    def _state_from_observation(
        self,
        observation: np.ndarray,
        reward: float = 0.0,
        terminated: bool = False,
        truncated: bool = False,
        info: dict | None = None,
        reset: bool = False,
    ) -> BWState:
        return BWState(
            observation=torch.from_numpy(observation),
            reward=reward,
            terminated=truncated,
            truncated=terminated,
            info={} if info is None else info,
            reset=reset,
        )

    def reset(self, seed: int | None = None) -> BWState:
        init_observation, init_info = self.env.reset(seed=seed)
        return self._state_from_observation(init_observation, info=init_info, reset=True)

    def render(self, state: BWState) -> np.ndarray | Tensor:
        return cast(np.ndarray, self.env.render())

    def sample_action(self) -> BWAction:
        env_sample = self.env.action_space.sample().tolist()
        return BWAction(
            hip_1=env_sample[0],
            knee_1=env_sample[1],
            hip_2=env_sample[2],
            knee_2=env_sample[3],
            log_prob=[0.0] * 4,
        )

    def step(self, action: BWAction) -> BWState:
        action_arr = np.array([action.hip_1, action.knee_1, action.hip_2, action.knee_2])
        observation_arr, reward, terminated, truncated, info = self.env.step(action_arr)
        return self._state_from_observation(observation_arr, float(reward), terminated, truncated, info)

    def terminated(self, state: BWState) -> bool:
        return cast(bool, state.terminated or state.truncated)


@dataclass
class RLDemoTaskConfig(ReinforcementLearningTaskConfig):
    hardcore: bool = conf_field(False, help="If set, use the hardcore environment")
    gamma: float = conf_field(0.99, help="The discount factor")
    gae_lmda: float = conf_field(0.9, help="The GAE factor (higher means more variance, lower means more bias)")
    clip: float = conf_field(0.16, help="The PPO clip factor")
    val_coef: float = conf_field(0.5, help="The value loss coefficient")
    ent_coef: float = conf_field(1e-2, help="The entropy coefficient")
    sample_clip_interval: int = conf_field(25, help="Sample a clip with this frequency")
    normalize_advantages: bool = conf_field(False, help="If set, normalize advantages")


Output = tuple[Tensor, Normal]
Loss = dict[str, Tensor]


@register_task("rl_demo", RLDemoTaskConfig)
class RLDemoTask(
    ReinforcementLearningTask[
        RLDemoTaskConfig,
        SimpleA2CModel,
        BWState,
        BWAction,
        Output,
        Loss,
    ],
):
    def __init__(self, config: RLDemoTaskConfig):
        super().__init__(config)

    def get_actions(self, model: SimpleA2CModel, states: list[BWState], optimal: bool) -> list[BWAction]:
        collated_states = self._device.recursive_apply(self.collate_fn(states))
        value = model.forward_value_net(collated_states.observation).cpu()
        p_dist = model.forward_policy_net(collated_states.observation)
        action = p_dist.mode if optimal else p_dist.sample()
        log_prob, action = p_dist.log_prob(action).cpu(), action.cpu()
        return [BWAction.from_policy(c, p, v) for c, p, v in zip(action.unbind(0), log_prob.unbind(0), value.unbind(0))]

    def get_environment(self) -> BipedalWalkerEnvironment:
        return BipedalWalkerEnvironment(hardcore=self.config.hardcore)

    def postprocess_trajectory(self, samples: list[tuple[BWState, BWAction]]) -> list[tuple[BWState, BWAction]]:
        def discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
            return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

        # Gets the reward-to-go for each timestep.
        rewards = np.array([cast(float, s.reward) for s, _ in samples])
        values = np.array([cast(float, a.value) for _, a in samples])

        # Computes the advantage estimate at each timestep.
        deltas = rewards[:-1] + self.config.gamma * values[1:] - values[:-1]
        advantages = discount_cumsum(deltas, self.config.gamma * self.config.gae_lmda)

        # Computes the reward-to-go at each timestep.
        returns = advantages + values[:-1]

        # Normalizes the advantages.
        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Adds the advantages and returns to the samples.
        for (_, sample), advantage, returns in zip(samples, advantages, returns):
            sample.advantage = advantage
            sample.returns = returns

        # Adds the last value to the samples.
        samples[-1][1].returns = values[-1]
        samples[-1][1].advantage = 0.0

        return samples

    def postprocess_trajectories(
        self,
        trajectories: list[list[tuple[BWState, BWAction]]],
    ) -> list[list[tuple[BWState, BWAction]]]:
        reward_arr = np.array([cast(float, s.reward) for t in trajectories for s, _ in t])
        self.logger.log_scalar("reward_mean", reward_arr.mean())
        self.logger.log_scalar("reward_std", reward_arr.std())
        return trajectories

    def run_model(self, model: SimpleA2CModel, batch: tuple[BWState, BWAction], state: State) -> Output:
        states, _ = batch
        obs = states.observation
        value = model.forward_value_net(obs)
        p_dist = model.forward_policy_net(obs)
        return value, p_dist

    def compute_loss(
        self,
        model: SimpleA2CModel,
        batch: tuple[BWState, BWAction],
        state: State,
        output: Output,
    ) -> Loss:
        _, actions = batch
        value, p_dist = output

        adv = cast(Tensor, actions.advantage)
        ret = cast(Tensor, actions.returns)
        old_log_probs = torch.cat(cast(list[Tensor], actions.log_prob), dim=-1)

        # Supervises the policy network.
        actions_tensor = actions.to_tensor().squeeze(2)  # (B, T, A)
        log_prob = p_dist.log_prob(actions_tensor)
        rt_theta = (log_prob - old_log_probs).exp()
        policy_loss = -torch.min(rt_theta * adv, rt_theta.clamp(1 - self.config.clip, 1 + self.config.clip) * adv)
        policy_loss = policy_loss.sum(1).mean()

        # Supervises the value network.
        value_loss = F.mse_loss(value, ret, reduction="mean")

        # Entropy loss to encourage exploration.
        entropy_loss = -p_dist.entropy().mean()

        # Logs additional metrics.
        self.logger.log_scalar("p_dist_std", lambda: p_dist.stddev.mean().item())
        if state.num_epoch_steps == 0 and state.num_epochs % self.config.sample_clip_interval == 0:
            self.logger.log_video("sample", self.sample_clip(model=model, use_tqdm=False))

        return {
            "policy": policy_loss,
            "value": value_loss * self.config.val_coef,
            "entropy": entropy_loss * self.config.ent_coef,
        }


def run_adhoc_test() -> None:
    """Runs adhoc tests for this task.

    Usage:
        python -m ml.tasks.rl_demo
    """

    configure_logging(use_tqdm=True)
    config = RLDemoTaskConfig()
    task = RLDemoTask(config)
    task.sample_clip(save_path="out/bipedal_walker.mp4", writer="opencv")


if __name__ == "__main__":
    run_adhoc_test()
