################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Any, List, Union

import torch
from omegaconf import ListConfig

from .mse_fn import LOSS_FN
from .registry import prepare_loss_function_kwargs


class NormalLoss(torch.nn.Module):
    def __init__(
        self,
        loss_property: List[str],
        loss_function_name: List[str],
        loss_property_weights: Union[List[float], None],
        loss_function_kwargs: Union[List[dict[str, Any]], None] = None,
        normalize: bool = False,
    ):
        super().__init__()
        assert isinstance(loss_property, (list, ListConfig)), (
            f"cfg.loss.loss_property should be a list, got {type(loss_property)}"
        )
        assert isinstance(loss_function_name, (list, ListConfig)), (
            f"cfg.loss.loss_function_name should be a list, got {type(loss_function_name)}"
        )
        assert isinstance(loss_property_weights, (list, ListConfig)), (
            f"cfg.loss.loss_property_weights should be a list, got {type(loss_property_weights)}"
        )
        assert len(loss_function_name) == len(loss_property_weights)
        assert len(loss_property) <= len(loss_function_name)
        self.loss_property = loss_property
        self.loss_function_name = loss_function_name
        self.loss_function_kwargs = prepare_loss_function_kwargs(
            loss_function_name, loss_function_kwargs
        )
        if normalize:
            normalizer = sum(loss_property_weights)
            self.loss_property_weights = [w / normalizer for w in loss_property_weights]
        else:
            self.loss_property_weights = loss_property_weights

    def forward(self, pred, label):
        total_loss = 0.0
        for i, func_name in enumerate(self.loss_function_name):
            loss = LOSS_FN[func_name](
                pred, label, **self.loss_function_kwargs[i]
            )
            total_loss += loss * self.loss_property_weights[i]
        return total_loss

    def __repr__(self):
        task_strs = [
            f"  - {p:<12} | weight={w:>7.3f} | fn={fn} | kwargs={kwargs}"
            for p, fn, w, kwargs in zip(
                self.loss_property,
                self.loss_function_name,
                self.loss_property_weights,
                self.loss_function_kwargs,
            )
        ]
        tasks_info = "\n".join(task_strs)
        return f"{self.__class__.__name__}(\n{tasks_info}\n)"
