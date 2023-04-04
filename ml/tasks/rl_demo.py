from dataclasses import dataclass
from typing import cast

import gymnasium as gym
import numpy as np
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

    @classmethod
    def from_policy(cls, policy: Tensor, log_prob: Tensor) -> "BWAction":
        assert policy.shape == (4,) and log_prob.shape == (4,)
        log_prob_list = log_prob.detach().cpu().tolist()
        return cls(policy[0].item(), policy[1].item(), policy[2].item(), policy[3].item(), log_prob_list)

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
    def __init__(self, hardcore: bool = False, seed: int = 1337) -> None:
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
    env_seed: int = conf_field(1337, help="The default environment initial seed")
    gamma: float = conf_field(0.99, help="The discount factor")
    lmda: float = conf_field(0.95, help="The GAE factor")
    clip: float = conf_field(0.2, help="The PPO clip factor")
    video_every_n_steps: int = conf_field(1000, help="The number of steps between video recordings")


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

    def get_actions(self, model: SimpleA2CModel, states: list[BWState]) -> list[BWAction]:
        collated_states = self._device.recursive_apply(self.collate_fn(states))
        p_dist = model.forward_policy_net(collated_states.observation)
        action = p_dist.sample()
        log_prob = p_dist.log_prob(action)
        return [BWAction.from_policy(c, p) for c, p in zip(action.unbind(0), log_prob.unbind(0))]

    def get_environment(self) -> BipedalWalkerEnvironment:
        return BipedalWalkerEnvironment(
            hardcore=self.config.hardcore,
            seed=self.config.env_seed,
        )

    def run_model(self, model: SimpleA2CModel, batch: tuple[BWState, BWAction], state: State) -> Output:
        states, _ = batch
        obs = states.observation
        value = model.forward_value_net(obs).squeeze(-1)
        p_dist = model.forward_policy_net(obs)
        return value, p_dist

    def compute_loss(
        self,
        model: SimpleA2CModel,
        batch: tuple[BWState, BWAction],
        state: State,
        output: Output,
    ) -> Loss:
        states, actions = batch
        value, p_dist = output
        old_log_prob = torch.cat(cast(list[Tensor], actions.log_prob), dim=-1)
        reward = cast(Tensor, states.reward).squeeze(-1)

        # Computes the advantage.
        target_v = reward[:, :-1] + self.config.gamma * value[:, 1:]
        adv = (target_v - value[:, :-1]).detach()

        # Supervises the value network.
        value_loss = F.mse_loss(value[:, :-1], target_v, reduction="none").sum(-1)  # (B)

        # Supervises the policy network.
        actions_tensor = actions.to_tensor().squeeze(2)  # (B, T, A)
        rt_theta = (p_dist.log_prob(actions_tensor) - old_log_prob).exp()
        adv, rt_theta = adv.unsqueeze(-1), rt_theta[:, :-1]
        policy_loss = -torch.min(rt_theta * adv, rt_theta.clamp(1 - self.config.clip, 1 + self.config.clip) * adv)
        policy_loss = policy_loss.flatten(1).sum(1)

        # Logs additional metrics.
        self.logger.log_scalar("reward", lambda: reward.mean().item())
        if state.num_steps % self.config.video_every_n_steps == 0:
            self.logger.log_video("sample", self.sample_clip(model=model, use_tqdm=False))

        return {
            "value": value_loss,
            "policy": policy_loss,
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
