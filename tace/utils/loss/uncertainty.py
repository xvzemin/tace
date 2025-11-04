################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import List, Optional


import torch
from torch import nn
from omegaconf import ListConfig


from .fn import LOSS_FN


class UncertaintyLoss(nn.Module):
    """
    Multi-task uncertainty weighted loss.
    Based on: "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    (https://arxiv.org/abs/1705.07115)
    """

    def __init__(
        self,
        loss_property: List[str],
        loss_function_name: List[str],
        loss_property_weights: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__()
        init_log_sigmas = loss_property_weights
        assert isinstance(
            loss_property, (List, ListConfig)
        ), f"``cfg.loss.loss_property`` should be a list, got {type(loss_property)}"
        assert isinstance(
            loss_function_name, (List, ListConfig)
        ), f"``cfg.loss.loss_function_name`` should be a list, got {type(loss_property)}"
        if init_log_sigmas is None:
            init_log_sigmas = [0.0] * len(loss_property)
        assert isinstance(
            init_log_sigmas, (List, ListConfig)
        ), f"``cfg.loss.loss_property_weights`` should be a list, got {type(loss_property)}"
        assert len(loss_property) == len(loss_function_name) == len(init_log_sigmas)
        for fn in loss_function_name:
            assert (
                fn in LOSS_FN
            ), f"{fn} not in LOSS_FN, add by yourself, all available function name are {list(LOSS_FN)}"

        self.loss_property = loss_property
        self.loss_function_name = loss_function_name
        self.log_sigmas = nn.ParameterDict()
        for p, val in zip(loss_property, init_log_sigmas):
            self.log_sigmas[p] = nn.Parameter(
                torch.tensor(val, dtype=torch.get_default_dtype())
            )

    def forward(self, pred, label):
        total_loss = 0.0
        for p, fn_name in zip(self.loss_property, self.loss_function_name):
            p_loss = LOSS_FN[fn_name](pred, label)
            log_sigma = self.log_sigmas[p]
            total_loss += 0.5 * torch.exp(-log_sigma) * p_loss + log_sigma
        return total_loss

    def __repr__(self):
        task_strs = [
            f"{p} (fn={fn}, log_sigma={v.item():.4f})"
            for p, fn, v in zip(
                self.loss_property, self.loss_function_name, self.log_sigmas.values()
            )
        ]
        tasks_info = ", ".join(task_strs)
        return f"{self.__class__.__name__}({tasks_info})"
