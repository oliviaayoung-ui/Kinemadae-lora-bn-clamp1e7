import torch


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}              # parameter EMA
        self.shadow_buffers = {}      # [NEW] buffer EMA (BN running_mean/var 등) — REPA-E 따라
        self.backup = {}
        self.backup_buffers = {}      # [NEW]

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        # [NEW] buffer register — float buffer 만 (BN running stats 등; int buffer 인 num_batches_tracked 등은 skip)
        for name, buf in self.model.named_buffers():
            if buf.dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
                self.shadow_buffers[name] = buf.data.clone()

    def update(self):
        # Parameter EMA (기존)
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)
        # [NEW] Buffer EMA — REPA-E update_ema 의 L92-95 과 동등
        for name, buf in self.model.named_buffers():
            if name in self.shadow_buffers:
                self.shadow_buffers[name].mul_(self.decay).add_(buf.data, alpha=1.0 - self.decay)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data
                param.data = self.shadow[name]
        # [NEW] buffer 도 apply
        for name, buf in self.model.named_buffers():
            if name in self.shadow_buffers:
                self.backup_buffers[name] = buf.data.clone()
                buf.data = self.shadow_buffers[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                param.data = self.backup[name]
        self.backup = {}
        # [NEW] buffer 도 restore
        for name, buf in self.model.named_buffers():
            if name in self.backup_buffers:
                buf.data = self.backup_buffers[name]
        self.backup_buffers = {}

    # [NEW] resume/load support: shadow + shadow_buffers 모두 직렬화
    def state_dict(self):
        return {'shadow': self.shadow, 'shadow_buffers': self.shadow_buffers}

    def load_state_dict(self, sd):
        # backward compat: 기존 ckpt 는 shadow 만 (dict 자체)
        if isinstance(sd, dict) and 'shadow' in sd and 'shadow_buffers' in sd:
            self.shadow = sd['shadow']
            self.shadow_buffers = sd['shadow_buffers']
        else:
            # legacy: dict 가 shadow 자체 → buffer 는 빈 dict (resume 시 register() 로 init)
            self.shadow = sd
            self.shadow_buffers = {}
