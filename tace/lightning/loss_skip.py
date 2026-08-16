################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


import logging

import torch
import torch.distributed as dist


class LossSkipController:
    """
    Skip controller with:
        - EMA using window size
        - dynamic threshold
        - manual threshold
        - warmup step
    """

    def __init__(
        self,
        manual_threshold=None,
        start_step=50000,
        ema_window=None,
        multiplier=None,
        skip_nan=True,
        skip_large=True,
    ):
        self.manual_threshold = manual_threshold
        self.start_step = start_step
        self.multiplier = multiplier
        self.skip_large = skip_large
        self.skip_nan = skip_nan

        self.ema_window = ema_window
        if ema_window is not None:
            if ema_window <= 0:
                raise ValueError("ema_window must be positive")
            self.ema_alpha = 1.0 / float(ema_window)
        else:
            self.ema_alpha = None

        self.loss_ema = None

    def _update_ema(self, loss_val: float):
        if self.ema_alpha is None:
            return

        if self.loss_ema is None:
            self.loss_ema = loss_val
        else:
            a = self.ema_alpha
            self.loss_ema = (1.0 - a) * self.loss_ema + a * loss_val

    def _get_dynamic_threshold(self):
        if self.loss_ema is None:
            return None
        return self.multiplier * self.loss_ema

    def _get_effective_threshold(self):
        dyn = self._get_dynamic_threshold()
        man = self.manual_threshold

        if man is not None and dyn is not None:
            return min(man, dyn)
        elif man is not None:
            return man
        elif dyn is not None:
            return dyn
        else:
            return None

    def check(self, loss: torch.Tensor, global_step: int):
        """
        Returns:
            skip: bool
            reason_code: int
            threshold: float or None

        reason_code:
            0 normal
            1 non-finite
            2 large
        """

        loss_detached = loss.detach()
        loss_finite = torch.isfinite(loss_detached)
        threshold = self._get_effective_threshold()

        reason_code = 0
        local_skip = False

        # (1) NaN / Inf always skip
        if (not loss_finite) and self.skip_nan:
            local_skip = True
            reason_code = 1

        # (2) Large loss
        elif self.skip_large:
            enable_large = (
                self.start_step is not None and global_step >= self.start_step
            )

            if enable_large and threshold is not None:
                if loss_detached.item() > threshold:
                    local_skip = True
                    reason_code = 2

        # ---- DDP sync ----
        if dist.is_available() and dist.is_initialized():
            device = loss.device
            reason_tensor = torch.tensor(reason_code, device=device, dtype=torch.int32)
            dist.all_reduce(reason_tensor, op=dist.ReduceOp.MAX)

            reason_code = int(reason_tensor.item())
            skip = reason_code > 0
        else:
            skip = local_skip

        # ---- EMA update (global decision) ----
        if (not skip) and loss_finite:
            self._update_ema(loss_detached.item())

        threshold = self._get_effective_threshold()

        return skip, reason_code, threshold

    def log_status(self, global_step):
        if self.loss_ema is None:
            return

        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() != 0:
                return

        dyn = self._get_dynamic_threshold()
        eff = self._get_effective_threshold()

        logging.info(
            f"[LossStats] step={global_step}, "
            f"window={self.ema_window}, "
            f"ema={self.loss_ema:.4f}, "
            f"dyn={dyn:.4f}, eff={eff:.4f}, "
            f"skip_large={self.skip_large}"
        )
