import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import bf16_ok, log


class RewardModel(nn.Module):
    """LLM backbone (LoRA) + scalar head, score = tanh(head(h_last)) in [-1, 1]."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last = attention_mask.sum(dim=1) - 1
        feat = h[torch.arange(h.size(0), device=h.device), last]
        return torch.tanh(self.head(feat.float())).squeeze(-1)


def build_reward_model(cfg, device):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(cfg["rm_model"], torch_dtype=torch.float32)
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    return RewardModel(get_peft_model(backbone, lora)).to(device)


class Encoder:
    def __init__(self, tokenizer, pb, max_len, device):
        self.tok, self.pb, self.max_len, self.device = tokenizer, pb, max_len, device
        self.tok.truncation_side = "left"
        self.tok.padding_side = "right"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def text(self, question, steps):
        return self.pb.with_steps(question, steps, open_next=False)

    def __call__(self, texts):
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=self.max_len)
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)


def _amp_dtype():
    return torch.bfloat16 if bf16_ok() else torch.float16


def train_reward_model(kind, items, enc, cfg, device):
    """kind='PQM': items = (text, q), MSE loss.
    kind='PPM': items = (pos_text, neg_text), pairwise Bradley-Terry loss."""
    model = build_reward_model(cfg, device)
    model.train()
    head = list(model.head.parameters())
    rest = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head")]
    opt = torch.optim.AdamW([{"params": rest, "lr": cfg["rm_lr"]},
                             {"params": head, "lr": cfg["rm_head_lr"]}], weight_decay=0.0)
    amp = _amp_dtype()
    scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)

    # PQM: 8 sequences per step; PPM: 4 pairs = 8 sequences per step
    bs = cfg["rm_batch"] if kind == "PQM" else max(1, cfg["rm_batch"] // 2)
    total = cfg["rm_train_steps"]
    if not items:
        log(f"  [{kind}] no training data, skipped")
        return model.eval(), [float("nan")]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / total))

    def stream():
        while True:
            data = list(items)
            random.shuffle(data)
            for b in range(0, len(data), bs):
                yield data[b:b + bs]

    batches, losses = stream(), []
    for step in range(1, total + 1):
        batch = next(batches)
        with torch.autocast(device_type="cuda", dtype=amp):
            if kind == "PQM":
                ids, mask = enc([t for t, _ in batch])
                target = torch.tensor([q for _, q in batch], device=device)
                loss = F.mse_loss(model(ids, mask), target)
            else:
                ids, mask = enc([p for p, _ in batch] + [n for _, n in batch])
                r = model(ids, mask)
                loss = -F.logsigmoid(r[:len(batch)] - r[len(batch):]).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        losses.append(loss.item())
        if step % 100 == 0 or step == total:
            log(f"  [{kind}] step {step}/{total} ({step * bs / len(items):.1f} passes) "
                f"loss {np.mean(losses[-100:]):.4f}")
    return model.eval(), losses


@torch.no_grad()
def score_texts(model, enc, texts, batch=32):
    out = []
    with torch.autocast(device_type="cuda", dtype=_amp_dtype()):
        for b in range(0, len(texts), batch):
            ids, mask = enc(texts[b:b + batch])
            out.extend(model(ids, mask).float().cpu().tolist())
    return out
