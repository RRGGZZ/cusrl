from collections.abc import Iterable
from dataclasses import dataclass

import torch

import cusrl
from cusrl.preset.optimizer import AdamFactory

__all__ = ["AgentFactory", "hook_suite"]


def hook_suite(
    normalize_observation: bool = True,
    gamma: float = 0.99,
    tau: float = 0.005,
    reward_scale: float = 1.0,
    target_entropy: float | None = None,
    alpha: float = 0.2,
    tune_alpha: bool = True,
    policy_delay: int = 1,
    max_grad_norm: float | None = 1.0,
) -> list[cusrl.template.Hook]:
    hooks = [
        cusrl.hook.ModuleInitialization(init_actor=False, init_critic=False),
        cusrl.hook.ObservationNormalization() if normalize_observation else None,
        cusrl.hook.FastSACLoss(
            gamma=gamma,
            tau=tau,
            reward_scale=reward_scale,
            target_entropy=target_entropy,
            alpha=alpha,
            tune_alpha=tune_alpha,
            policy_delay=policy_delay,
        ),
        cusrl.hook.GradientClipping(max_grad_norm) if max_grad_norm is not None else None,
    ]
    return [hook for hook in hooks if hook is not None]


@dataclass
class AgentFactory(cusrl.template.ActorCritic.Factory):
    num_steps_per_update: int = 1
    actor_hidden_dims: Iterable[int] = (256, 256)
    critic_hidden_dims: Iterable[int] = (256, 256)
    activation_fn: str | type[torch.nn.Module] = "ReLU"
    lr: float = 3e-4
    replay_batches: int = 1
    replay_batch_size: int = 256
    normalize_observation: bool = True
    gamma: float = 0.99
    tau: float = 0.005
    reward_scale: float = 1.0
    target_entropy: float | None = None
    alpha: float = 0.2
    tune_alpha: bool = True
    policy_delay: int = 1
    max_grad_norm: float | None = 1.0
    device: str | torch.device | None = None
    compile: bool = False
    autocast: bool | None | str | torch.dtype = False

    def __post_init__(self):
        super().__init__(
            num_steps_per_update=self.num_steps_per_update,
            actor_factory=cusrl.Actor.Factory(
                backbone_factory=cusrl.Mlp.Factory(
                    hidden_dims=self.actor_hidden_dims,
                    activation_fn=self.activation_fn,
                    ends_with_activation=True,
                ),
                distribution_factory=cusrl.NormalDist.Factory(),
            ),
            critic_factory=cusrl.Value.Factory(
                backbone_factory=cusrl.Mlp.Factory(
                    hidden_dims=self.critic_hidden_dims,
                    activation_fn=self.activation_fn,
                    ends_with_activation=True,
                ),
                action_aware=True,
            ),
            optimizer_factory=AdamFactory(defaults={"lr": self.lr}),
            sampler=cusrl.AutoRandomSampler(
                num_batches=self.replay_batches,
                batch_size=self.replay_batch_size,
            ),
            hooks=hook_suite(
                normalize_observation=self.normalize_observation,
                gamma=self.gamma,
                tau=self.tau,
                reward_scale=self.reward_scale,
                target_entropy=self.target_entropy,
                alpha=self.alpha,
                tune_alpha=self.tune_alpha,
                policy_delay=self.policy_delay,
                max_grad_norm=self.max_grad_norm,
            ),
            device=self.device,
            compile=self.compile,
            autocast=self.autocast,
        )
