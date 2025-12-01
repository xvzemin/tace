################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import List, Optional


import torch
from omegaconf import ListConfig


from .fn import LOSS_FN


class NormalLoss(torch.nn.Module):
    def __init__(
        self,
        loss_property: List[str],
        loss_function_name: List[str],
        loss_property_weights: Optional[List[float]],
        normalize: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert isinstance(
            loss_property, (List, ListConfig)
        ), f"``cfg.loss.loss_property`` should be a list, got {type(loss_property)}"
        assert isinstance(
            loss_function_name, (List, ListConfig)
        ), f"``cfg.loss.loss_function_name`` should be a list, got {type(loss_property)}"
        if loss_property_weights is None:
            loss_property_weights = [0.0] * len(loss_property)
        assert isinstance(
            loss_property_weights, (List, ListConfig)
        ), f"``cfg.loss.loss_property_weights`` should be a list, got {type(loss_property)}"
        assert (
            len(loss_property) == len(loss_function_name) == len(loss_property_weights)
        )
        for fn in loss_function_name:
            assert (
                fn in LOSS_FN
            ), f"{fn} not in LOSS_FN, add by yourself, all available function name are {list(LOSS_FN)}"
        self.loss_property = loss_property
        self.loss_function_name = loss_function_name
        if normalize:
            normalizer = sum(loss_property_weights)
            self.loss_property_weights = [w / normalizer for w in loss_property_weights]
        else:
            self.loss_property_weights = loss_property_weights

    def forward(self, pred, label):
        total_loss = 0.0
        for i, func_name in enumerate(self.loss_function_name):
            loss = LOSS_FN[func_name](pred, label)
            total_loss += loss * self.loss_property_weights[i]
        return total_loss

    def __repr__(self):
        task_strs = [
            f"{p}: {fn} (weight={w:.4f})"
            for p, fn, w in zip(
                self.loss_property, self.loss_function_name, self.loss_property_weights
            )
        ]
        tasks_info = ", ".join(task_strs)
        return f"{self.__class__.__name__}({tasks_info})"
