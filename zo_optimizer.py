from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class ZeroOrderOptimizer:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-2,
        eps: float = 1e-4,
        perturbation_mode: str = "rademacher",
    ) -> None:
        self.model = model
        self.lr = lr
        self.eps = eps
        self.perturbation_mode = perturbation_mode

        self.layer_names = [
            "fc.weight",
            "fc.bias",
            "layer4.1.bn2.weight",
            "layer4.1.bn2.bias",
        ]

        self.step_idx = 0
        self.beta = 0.9
        self.momentum: dict[str, torch.Tensor] = {}

    def _set_bn_eval(self) -> None:
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def _active_params(self) -> dict[str, nn.Parameter]:
        named_params = dict(self.model.named_parameters())

        missing = [name for name in self.layer_names if name not in named_params]
        if missing:
            raise KeyError(f"Missing parameters: {missing}")

        return {name: named_params[name] for name in self.layer_names}

    def _sample_direction(self, param: torch.Tensor) -> torch.Tensor:
        direction = torch.empty_like(param).bernoulli_(0.5).mul_(2.0).sub_(1.0)

        direction_norm = direction.norm()
        if direction_norm > 0:
            direction = direction / direction_norm

        return direction

    def _maybe_update_layers(self) -> None:
        if self.step_idx == 80:
            candidates = [
                "fc.weight",
                "fc.bias",
                "layer4.1.bn2.weight",
                "layer4.1.bn2.bias",
            ]

            named_params = dict(self.model.named_parameters())
            self.layer_names = [name for name in candidates if name in named_params]

    def _estimate_grad(
        self,
        loss_fn: Callable[[], float],
        params: dict[str, nn.Parameter],
    ) -> tuple[dict[str, torch.Tensor], float]:
        directions = {
            name: self._sample_direction(param)
            for name, param in params.items()
        }

        with torch.no_grad():
            for name, param in params.items():
                param.add_(self.eps * directions[name])

            f_plus = loss_fn()

            for name, param in params.items():
                param.sub_(2.0 * self.eps * directions[name])

            f_minus = loss_fn()

            for name, param in params.items():
                param.add_(self.eps * directions[name])

        scale = (f_plus - f_minus) / (2.0 * self.eps)

        grads = {
            name: scale * directions[name]
            for name in params
        }

        approx_loss = 0.5 * (f_plus + f_minus)
        return grads, float(approx_loss)

    def _update_params(
        self,
        params: dict[str, nn.Parameter],
        grads: dict[str, torch.Tensor],
    ) -> None:
        with torch.no_grad():
            for name, param in params.items():
                if name not in self.momentum:
                    self.momentum[name] = torch.zeros_like(param)

                self.momentum[name].mul_(self.beta).add_(
                    grads[name],
                    alpha=1.0 - self.beta,
                )

                update = self.momentum[name]

                update_norm = update.norm()
                if update_norm > 0:
                    update = update / (update_norm + 1e-8)

                param.sub_(self.lr * update)

    def step(self, loss_fn: Callable[[], float]) -> float:
        self._set_bn_eval()
        self._maybe_update_layers()

        params = self._active_params()
        grads, approx_loss = self._estimate_grad(loss_fn, params)
        self._update_params(params, grads)

        self.step_idx += 1
        return approx_loss