import json
import os

import torch


class MatrixGame3DebugDumper:
    def __init__(self, enabled, dump_dir, max_steps, rank):
        self.enabled = bool(enabled and rank == 0)
        self.dump_dir = os.path.abspath(dump_dir) if dump_dir is not None else None
        self.max_steps = int(max_steps)
        self.global_step = 0
        self._limit_logged = False
        self._summary_fp = None

        if self.enabled:
            if self.dump_dir is None:
                raise ValueError("debug_dump_dir must be set when debug dump is enabled.")
            os.makedirs(self.dump_dir, exist_ok=True)
            self._summary_fp = open(
                os.path.join(self.dump_dir, "steps_summary.jsonl"),
                "w",
                encoding="utf-8",
            )
            print(
                f"[MG3_DEBUG] Enabled. dump_dir={self.dump_dir} max_steps={self.max_steps}",
                flush=True,
            )

    @classmethod
    def from_args(cls, args, rank):
        return cls(
            enabled=(args is not None and getattr(args, "debug_dump_steps", False)),
            dump_dir=(None if args is None else getattr(args, "debug_dump_dir", None)),
            max_steps=(50 if args is None else getattr(args, "debug_dump_max_steps", 50)),
            rank=rank,
        )

    def close(self):
        if self._summary_fp is not None:
            self._summary_fp.close()
            self._summary_fp = None

    def should_dump(self):
        if not self.enabled:
            return False
        if self.global_step < self.max_steps:
            return True
        if not self._limit_logged:
            print(
                f"[MG3_DEBUG] Reached debug_dump_max_steps={self.max_steps}; skipping remaining diffusion steps.",
                flush=True,
            )
            self._limit_logged = True
        return False

    def dump_step(
        self,
        *,
        clip_index,
        clip_step_index,
        timestep,
        latent_before,
        noise_pred,
        latent_after,
        fixed_latent_frames,
        mask=None,
    ):
        if not self.should_dump():
            return

        step_index = self.global_step
        step_prefix = f"step_{step_index:03d}"

        self._save_tensor(f"{step_prefix}_latent_before.pt", latent_before)
        self._save_tensor(f"{step_prefix}_noise_pred.pt", noise_pred)
        self._save_tensor(f"{step_prefix}_latent_after.pt", latent_after)

        record = {
            "step_index": step_index,
            "clip_index": int(clip_index),
            "clip_step_index": int(clip_step_index),
            "timestep": float(timestep.detach().item()) if torch.is_tensor(timestep) else float(timestep),
            "fixed_latent_frames": int(fixed_latent_frames),
            "mask": None if mask is None else self._tensor_summary(mask),
            "latent_before": self._tensor_summary(latent_before),
            "noise_pred": self._tensor_summary(noise_pred),
            "latent_after": self._tensor_summary(latent_after),
        }

        line = json.dumps(record, ensure_ascii=False)
        self._summary_fp.write(line + "\n")
        self._summary_fp.flush()
        print(f"[MG3_DEBUG] {line}", flush=True)

        self.global_step += 1

    def _save_tensor(self, filename, tensor):
        torch.save(tensor.detach().cpu(), os.path.join(self.dump_dir, filename))

    def _tensor_summary(self, tensor):
        tensor_detached = tensor.detach()
        flat = tensor_detached.reshape(-1)

        if flat.numel() == 0:
            return {
                "shape": list(tensor_detached.shape),
                "dtype": str(tensor_detached.dtype),
                "device": str(tensor_detached.device),
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "head": [],
            }

        return {
            "shape": list(tensor_detached.shape),
            "dtype": str(tensor_detached.dtype),
            "device": str(tensor_detached.device),
            "min": tensor_detached.min().item(),
            "max": tensor_detached.max().item(),
            "mean": tensor_detached.mean().item(),
            "std": tensor_detached.std(unbiased=False).item() if flat.numel() > 1 else 0.0,
            "head": flat[:8].cpu().tolist(),
        }
