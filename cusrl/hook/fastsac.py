import copy
from typing import cast

import torch
from torch import Tensor, nn

from cusrl.template import ActorCritic, Hook
from cusrl.utils.dict_utils import get_first

__all__ = ["FastSACLoss"]


class FastSACLoss(Hook[ActorCritic]):
    """Implements a compact Soft Actor-Critic style objective.

    This hook trains:
    - a policy (agent.actor),
    - two Q-networks (agent.critic as Q1 and an auxiliary Q2),
    - optional entropy temperature (alpha) via automatic tuning.

    Notes:
        - This hook assumes a continuous action policy with reparameterized
          sampling (e.g. ``NormalDist``).
        - It is intended to be used together with a replay-like random sampler.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        tau: float = 0.005,
        reward_scale: float = 1.0,
        target_entropy: float | None = None,
        alpha: float = 0.2,
        tune_alpha: bool = True,
        policy_delay: int = 1,
    ):
        if gamma < 0 or gamma >= 1:
            raise ValueError("'gamma' should be in [0, 1).")
        if tau <= 0 or tau > 1:
            raise ValueError("'tau' should be in (0, 1].")
        if alpha <= 0:
            raise ValueError("'alpha' should be positive.")
        if policy_delay <= 0:
            raise ValueError("'policy_delay' should be positive.")
        super().__init__()

        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale
        self.target_entropy = target_entropy
        self.tune_alpha = tune_alpha
        self.policy_delay = policy_delay

        # Mutable attributes
        self.alpha: float = alpha
        self.register_mutable("alpha")

        # Runtime attributes
        self.q2: nn.Module
        self.target_q1: nn.Module
        self.target_q2: nn.Module
        self.temperature: nn.Module
        self.mse_loss: nn.MSELoss
        self._optim_step_index: int

    def init(self):
        if not self.agent.critic.action_aware:
            raise ValueError("FastSACLoss requires an action-aware critic (Q-function).")

        self.register_module("q2", self.agent.critic_factory(self.agent.state_dim + self.agent.action_dim, self.agent.value_dim))
        self.register_module("target_q1", copy.deepcopy(self.agent.critic))
        self.register_module("target_q2", copy.deepcopy(self.q2))
        self.target_q1.requires_grad_(False)
        self.target_q2.requires_grad_(False)

        class Temperature(nn.Module):
            def __init__(self, init_log_alpha: Tensor, tune: bool):
                super().__init__()
                self.log_alpha = nn.Parameter(init_log_alpha, requires_grad=tune)

            def forward(self):
                return self.log_alpha.exp()

        self.register_module(
            "temperature",
            Temperature(self.agent.to_tensor(float(self.alpha)).log(), self.tune_alpha),
        )
        self.mse_loss = nn.MSELoss()
        self._optim_step_index = 0

    def objective(self, batch):
        actor = self.agent.actor
        q1 = self.agent.critic
        q2 = self.q2
        target_q1 = self.target_q1
        target_q2 = self.target_q2

        observation = cast(Tensor, batch["observation"])
        next_observation = cast(Tensor, batch["next_observation"])
        state = cast(Tensor, get_first(batch, "state", "observation"))
        next_state = cast(Tensor, get_first(batch, "next_state", "next_observation"))
        action = cast(Tensor, batch["action"])
        reward = cast(Tensor, batch["reward"])
        done = cast(Tensor, batch["done"]).to(reward.dtype)

        with self.agent.autocast(), torch.no_grad():
            next_dist, _ = actor(next_observation, done=done.bool())
            next_action, next_logp = actor.distribution.sample_from_dist(next_dist)
            target_q = torch.min(
                target_q1.evaluate(next_state, action=next_action),
                target_q2.evaluate(next_state, action=next_action),
            ) - self.get_alpha().detach() * next_logp
            td_target = reward * self.reward_scale + (1.0 - done) * self.gamma * target_q

        with self.agent.autocast():
            q1_pred = q1.evaluate(state, action=action)
            q2_pred = q2.evaluate(state, action=action)
            critic_loss = self.mse_loss(q1_pred, td_target) + self.mse_loss(q2_pred, td_target)

        self.agent.record(q1=q1_pred, q2=q2_pred, critic_loss=critic_loss)

        update_policy = self._optim_step_index % self.policy_delay == 0
        actor_loss = None
        alpha_loss = None
        if update_policy:
            with self.agent.autocast():
                policy_dist, _ = actor(observation, memory=batch.get("actor_memory"), done=done.bool())
                policy_action, policy_logp = actor.distribution.sample_from_dist(policy_dist)
                min_q_pi = torch.min(q1.evaluate(state, action=policy_action), q2.evaluate(state, action=policy_action))
                alpha = self.get_alpha()
                actor_loss = (alpha.detach() * policy_logp - min_q_pi).mean()

                if self.tune_alpha:
                    target_entropy = (
                        self.target_entropy if self.target_entropy is not None else -float(self.agent.action_dim)
                    )
                    alpha_loss = -(self.temperature.log_alpha * (policy_logp + target_entropy).detach()).mean()

            self.agent.record(actor_loss=actor_loss, alpha=alpha)
            if alpha_loss is not None:
                self.agent.record(alpha_loss=alpha_loss)

        total_loss = critic_loss
        if actor_loss is not None:
            total_loss = total_loss + actor_loss
        if alpha_loss is not None:
            total_loss = total_loss + alpha_loss
        return total_loss

    @torch.no_grad()
    def post_optim(self):
        self._optim_step_index += 1
        self._soft_update(self.target_q1, self.agent.critic, self.tau)
        self._soft_update(self.target_q2, self.q2, self.tau)
        self.alpha = float(self.get_alpha().detach().item())

    def get_alpha(self) -> Tensor:
        log_alpha = self.temperature.log_alpha
        if self.tune_alpha:
            return self.temperature()
        return log_alpha.detach().exp()

    @staticmethod
    @torch.no_grad()
    def _soft_update(target: nn.Module, source: nn.Module, tau: float):
        for target_param, source_param in zip(target.parameters(), source.parameters(), strict=True):
            target_param.data.mul_(1 - tau).add_(source_param.data, alpha=tau)

