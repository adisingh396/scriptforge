"""Exponential moving average of model weights."""

import copy
import torch


class ModelEma:
    def __init__(self, model=None, base_decay=0.999):
        self.base_decay = base_decay
        self.num_updates = 0
        if model is not None:
            self.model = copy.deepcopy(model).eval()
            self._device = next(model.parameters()).device
            self.model.to(self._device)
        else:
            self.model = None

    def _decay(self):
        t = min(1.0, self.num_updates / 1000.0)
        return 0.9 + (self.base_decay - 0.9) * t

    @torch.no_grad()
    def update(self, model):
        if self.model is None:
            self.model = copy.deepcopy(model).eval()
        self.num_updates += 1
        decay = self._decay()
        # params
        for p_ema, p in zip(self.model.parameters(), model.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
        # buffers
        for b_ema, b in zip(self.model.buffers(), model.buffers()):
            b_ema.data.copy_(b.data)

    def copy_to(self, model):
        if self.model is not None:
            model.load_state_dict(self.model.state_dict())

    def state_dict(self):
        if self.model is None:
            return {}
        sd = self.model.state_dict()
        sd["_num_updates"] = torch.tensor(self.num_updates)
        return sd

    def load_state_dict(self, sd):
        if not sd:
            return
        nu = sd.pop("_num_updates", None)
        if self.model is not None:
            self.model.load_state_dict(sd, strict=False)
        if nu is not None:
            try:
                self.num_updates = int(nu.item()) if torch.is_tensor(nu) else int(nu)
            except Exception:
                pass
