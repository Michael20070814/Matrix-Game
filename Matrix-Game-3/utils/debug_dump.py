import json
import os
from typing import Any

import torch


def _matrix_game3_basic_tensor_probe(tensor: torch.Tensor, head_values: int = 8) -> dict[str, Any]:
    tensor = tensor.detach()
    tensor_fp32 = tensor.to(dtype=torch.float32)
    flattened = tensor_fp32.reshape(-1)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "min": float(tensor_fp32.min().item()),
        "max": float(tensor_fp32.max().item()),
        "mean": float(tensor_fp32.mean().item()),
        "std": float(tensor_fp32.std(unbiased=False).item()),
        "head": flattened[:head_values].cpu().tolist(),
    }


def _matrix_game3_forward_value_probe(value: Any) -> Any:
    if value is None:
        return None
    if torch.is_tensor(value):
        return _matrix_game3_basic_tensor_probe(value)
    if isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "len": len(value),
            "head": [_matrix_game3_forward_value_probe(item) for item in value[:8]],
        }
    if isinstance(value, dict):
        return {k: _matrix_game3_forward_value_probe(v) for k, v in value.items()}
    return str(value)


class MatrixGame3DebugDumper:
    _FORWARD_KEYS = (
        "x",
        "t",
        "context",
        "seq_len",
        "mouse_cond",
        "keyboard_cond",
        "x_memory",
        "timestep_memory",
        "mouse_cond_memory",
        "keyboard_cond_memory",
        "plucker_emb",
        "memory_latent_idx",
        "predict_latent_idx",
    )

    def __init__(self, enabled, dump_dir, step_filter, max_steps, rank):
        self.enabled = bool(enabled and rank == 0)
        self.dump_dir = os.path.abspath(dump_dir) if dump_dir is not None else None
        self.max_steps = int(max_steps)
        self.step_filter = self._parse_step_filter(step_filter)
        self._summary_fp = None
        self._max_steps_logged = False

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
                f"[MG3_DEBUG] Enabled. dump_dir={self.dump_dir} max_steps={self.max_steps} step_filter={None if self.step_filter is None else sorted(self.step_filter)}",
                flush=True,
            )

    @classmethod
    def from_args(cls, args, rank):
        return cls(
            enabled=(args is not None and getattr(args, "debug_dump_steps", False)),
            dump_dir=(None if args is None else getattr(args, "debug_dump_dir", None)),
            step_filter=(None if args is None else getattr(args, "debug_dump_step_filter", None)),
            max_steps=(50 if args is None else getattr(args, "debug_dump_max_steps", 50)),
            rank=rank,
        )

    def close(self):
        if self._summary_fp is not None:
            self._summary_fp.close()
            self._summary_fp = None

    def should_dump_step(self, step_index: int) -> bool:
        if not self.enabled:
            return False
        if step_index >= self.max_steps:
            if not self._max_steps_logged:
                print(
                    f"[MG3_DEBUG] Reached debug_dump_max_steps={self.max_steps}; skipping later diffusion steps.",
                    flush=True,
                )
                self._max_steps_logged = True
            return False
        if self.step_filter is not None and step_index not in self.step_filter:
            return False
        return True

    def probe_forward_kwargs(self, forward_kwargs: dict[str, Any]) -> dict[str, Any]:
        return {
            key: _matrix_game3_forward_value_probe(forward_kwargs.get(key))
            for key in self._FORWARD_KEYS
        }

    def build_mask(self, total_latent_frames: int, fixed_latent_frames: int, device, dtype) -> torch.Tensor:
        mask = torch.zeros((total_latent_frames,), device=device, dtype=dtype)
        mask[fixed_latent_frames:] = 1
        return mask

    def dump_step(
        self,
        *,
        step_index: int,
        timestep,
        fixed_latent_frames: int,
        mask: torch.Tensor,
        latent_before: torch.Tensor,
        noise_pred_cond: torch.Tensor | None,
        noise_pred_uncond: torch.Tensor | None,
        noise_pred_guided: torch.Tensor | None,
        noise_pred: torch.Tensor | None,
        latent_after: torch.Tensor,
        forward_kwargs_cond: dict[str, Any] | None,
        forward_kwargs_uncond: dict[str, Any] | None,
    ):
        if not self.should_dump_step(step_index):
            return

        step_payload = {
            "step_index": step_index,
            "timestep": float(timestep.detach().item()) if torch.is_tensor(timestep) else float(timestep),
            "fixed_latent_frames": int(fixed_latent_frames),
            "mask": _matrix_game3_basic_tensor_probe(mask),
            "latent_before": _matrix_game3_basic_tensor_probe(latent_before),
            "noise_pred_cond": _matrix_game3_basic_tensor_probe(noise_pred_cond) if noise_pred_cond is not None else None,
            "noise_pred_uncond": _matrix_game3_basic_tensor_probe(noise_pred_uncond) if noise_pred_uncond is not None else None,
            "noise_pred_guided": _matrix_game3_basic_tensor_probe(noise_pred_guided) if noise_pred_guided is not None else None,
            "noise_pred": _matrix_game3_basic_tensor_probe(noise_pred) if noise_pred is not None else None,
            "latent_after": _matrix_game3_basic_tensor_probe(latent_after),
            "forward_kwargs_cond": forward_kwargs_cond,
            "forward_kwargs_uncond": forward_kwargs_uncond,
        }

        self._save_tensor(f"step_{step_index:03d}_latent_before.pt", latent_before)
        if noise_pred_cond is not None:
            self._save_tensor(f"step_{step_index:03d}_noise_pred_cond.pt", noise_pred_cond)
        if noise_pred_uncond is not None:
            self._save_tensor(f"step_{step_index:03d}_noise_pred_uncond.pt", noise_pred_uncond)
        if noise_pred_guided is not None:
            self._save_tensor(f"step_{step_index:03d}_noise_pred_guided.pt", noise_pred_guided)
        if noise_pred is not None:
            self._save_tensor(f"step_{step_index:03d}_noise_pred.pt", noise_pred)
        self._save_tensor(f"step_{step_index:03d}_latent_after.pt", latent_after)

        line = json.dumps(step_payload, ensure_ascii=False)
        self._summary_fp.write(line + "\n")
        self._summary_fp.flush()
        print(f"[MG3_DEBUG] {line}", flush=True)

    def _save_tensor(self, filename: str, tensor: torch.Tensor):
        torch.save(tensor.detach().cpu(), os.path.join(self.dump_dir, filename))

    def _parse_step_filter(self, step_filter: str | None):
        if step_filter is None:
            return None
        parsed = set()
        for raw_step in step_filter.split(","):
            raw_step = raw_step.strip()
            if not raw_step:
                continue
            step_index = int(raw_step)
            if step_index < 0:
                raise ValueError("debug_dump_step_filter only supports non-negative step indices.")
            parsed.add(step_index)
        return parsed if parsed else None
