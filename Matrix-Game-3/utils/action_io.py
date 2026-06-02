"""
Action-file (replay) input support for Matrix-Game-3.

This module loads pre-defined action sequences from a JSON file and converts
them into the ``keyboard_condition`` ([T, 6]) and ``mouse_condition`` ([T, 2])
tensors consumed by both the non-interactive and interactive inference
pipelines. When an action file is provided, random action generation
(``utils.conditions.Bench_actions_universal`` via ``utils.utils.get_data``)
and manual ``input()`` collection (``get_current_action``) are bypassed.

Two JSON schemas are supported:

Schema A -- Tensor format (frame-level)::

    {
      "keyboard_condition": [[0,0,1,0,0,0], ...],   # shape [T, 6]
      "mouse_condition":    [[0.0,-0.1], ...]        # shape [T, 2]
    }

Schema B -- Clip format (token-level)::

    # One entry per diffusion iteration (default, clip count == num_iterations):
    {
      "clips": [
        {"mouse": "u", "keyboard": "w"},
        {"mouse": "j", "keyboard": "a"},
        ...
      ]
    }

    # Uniform sub-clip granularity (unit_frames frames per entry):
    {
      "unit_frames": 20,
      "clips": [
        {"mouse": "u", "keyboard": "w"},
        ...
      ]
    }

    # Explicit per-clip frame counts (frames must sum to total):
    {
      "clips": [
        {"mouse": "u", "keyboard": "w", "frames": 57},
        {"mouse": "j", "keyboard": "a", "frames": 40},
        ...
      ]
    }

``unit_frames`` can also be supplied on the command line via
``--control_unit_frames``, in which case the file value takes precedence.

The token maps below are copied verbatim from
``pipeline.inference_interactive_pipeline.get_current_action`` so replay actions
match interactive actions exactly.

VIDEO_FPS note:
    The output video runs at 40 fps.  0.5 s = 20 frames (``unit_frames=20``).
    Other common values: 10 frames (0.25 s), 40 frames (1 s).
"""

import json
import math
import os

import numpy as np
import torch

# --- Frame schedule (must match the inference pipelines) ---------------------
FIRST_CLIP_FRAMES = 57
SUBSEQUENT_CLIP_FRAMES = 40

# --- Output frame rate (used only for documentation / CLI hints) --------------
VIDEO_FPS = 40  # frames per second of the generated video

# --- Condition dimensions (must match the Action Module) ---------------------
KEYBOARD_DIM = 6
MOUSE_DIM = 2

# --- Token maps (extracted from get_current_action) --------------------------
CAM_VALUE = 0.1
MOUSE_TOKEN_MAP = {
    "i": [CAM_VALUE, 0.0],   # camera up
    "k": [-CAM_VALUE, 0.0],  # camera down
    "j": [0.0, -CAM_VALUE],  # camera left
    "l": [0.0, CAM_VALUE],   # camera right
    "u": [0.0, 0.0],         # no camera move
}
KEYBOARD_TOKEN_MAP = {
    "w": [1, 0, 0, 0, 0, 0],  # forward
    "s": [0, 1, 0, 0, 0, 0],  # back
    "a": [0, 0, 1, 0, 0, 0],  # left
    "d": [0, 0, 0, 1, 0, 0],  # right
    "q": [0, 0, 0, 0, 0, 0],  # no movement
}

# Defaults applied when a clip omits a key (both are "no-op" tokens).
DEFAULT_MOUSE_TOKEN = "u"
DEFAULT_KEYBOARD_TOKEN = "q"


class ActionFileError(ValueError):
    """Raised for any problem loading or validating an action file."""


def expected_total_frames(num_iterations):
    """Total frame count for ``num_iterations`` = 57 + (num_iterations - 1) * 40."""
    return FIRST_CLIP_FRAMES + (num_iterations - 1) * SUBSEQUENT_CLIP_FRAMES


def default_clip_frames(num_iterations):
    """Default per-iteration frame schedule: [57, 40, 40, ...]."""
    if num_iterations < 1:
        raise ActionFileError(f"num_iterations must be >= 1, got {num_iterations}.")
    return [FIRST_CLIP_FRAMES] + [SUBSEQUENT_CLIP_FRAMES] * (num_iterations - 1)


def unit_clip_frames(unit_frames, total):
    """Frame schedule for uniform sub-clip granularity.

    Each entry covers ``unit_frames`` frames; the last entry absorbs the
    remainder so the total is always exactly ``total``.

    Returns a list of per-entry frame counts of length
    ``ceil(total / unit_frames)``.
    """
    n = math.ceil(total / unit_frames)
    frames = [unit_frames] * (n - 1) + [total - (n - 1) * unit_frames]
    return frames


def expected_clip_count(unit_frames, total):
    """Number of clips needed when using ``unit_frames``."""
    return math.ceil(total / unit_frames)


def _supported_tokens_msg():
    return (
        f"Supported mouse tokens: {sorted(MOUSE_TOKEN_MAP)} "
        f"(i=up, k=down, j=left, l=right, u=no-move). "
        f"Supported keyboard tokens: {sorted(KEYBOARD_TOKEN_MAP)} "
        f"(w=forward, s=back, a=left, d=right, q=no-move)."
    )


def _load_json(actions_file):
    if not os.path.isfile(actions_file):
        raise ActionFileError(f"Actions file not found: {actions_file!r}")
    try:
        with open(actions_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ActionFileError(
            f"Failed to parse JSON actions file {actions_file!r}: {exc}"
        )
    except OSError as exc:
        raise ActionFileError(f"Could not read actions file {actions_file!r}: {exc}")
    if not isinstance(data, dict):
        raise ActionFileError(
            "Actions file must contain a JSON object at the top level "
            "(with a 'clips' list or 'keyboard_condition'/'mouse_condition' arrays)."
        )
    return data


def _parse_tensor_format(data, num_iterations, total):
    if "keyboard_condition" not in data or "mouse_condition" not in data:
        raise ActionFileError(
            "Tensor-format actions file must contain both 'keyboard_condition' "
            "and 'mouse_condition'."
        )
    try:
        keyboard = torch.tensor(data["keyboard_condition"], dtype=torch.float32)
        mouse = torch.tensor(data["mouse_condition"], dtype=torch.float32)
    except (TypeError, ValueError) as exc:
        raise ActionFileError(
            f"Could not convert 'keyboard_condition'/'mouse_condition' to tensors: {exc}"
        )

    if keyboard.ndim != 2 or keyboard.shape[1] != KEYBOARD_DIM:
        raise ActionFileError(
            f"keyboard_condition must have shape [T, {KEYBOARD_DIM}], "
            f"got {list(keyboard.shape)}."
        )
    if mouse.ndim != 2 or mouse.shape[1] != MOUSE_DIM:
        raise ActionFileError(
            f"mouse_condition must have shape [T, {MOUSE_DIM}], "
            f"got {list(mouse.shape)}."
        )
    if keyboard.shape[0] != mouse.shape[0]:
        raise ActionFileError(
            "keyboard_condition and mouse_condition must share the same T, got "
            f"T={keyboard.shape[0]} (keyboard) vs T={mouse.shape[0]} (mouse)."
        )

    t = keyboard.shape[0]
    if t != total:
        raise ActionFileError(
            f"Frame-count mismatch: actions file has T={t}, but num_iterations="
            f"{num_iterations} requires T={total} (= 57 + ({num_iterations} - 1) * 40)."
        )
    return keyboard, mouse, default_clip_frames(num_iterations)


def _resolve_clip_frames(clips, num_iterations, total, unit_frames=None):
    """Determine per-clip frame counts using one of three modes.

    Priority (highest to lowest):
    1. Per-clip ``frames`` field in each clip object.
    2. ``unit_frames`` (from the top-level JSON field or CLI ``--control_unit_frames``).
    3. Default: one clip per diffusion iteration (57 / 40 schedule).
    """
    frames_present = ["frames" in c for c in clips]

    # --- Mode 1: explicit per-clip frames ------------------------------------
    if any(frames_present):
        if unit_frames is not None:
            raise ActionFileError(
                "Cannot mix 'unit_frames' (top-level) with per-clip 'frames'. "
                "Use one or the other."
            )
        if not all(frames_present):
            raise ActionFileError(
                "If any clip specifies 'frames', every clip must specify 'frames'."
            )
        clip_frames = []
        for i, clip in enumerate(clips):
            f = clip["frames"]
            if not isinstance(f, int) or isinstance(f, bool) or f <= 0:
                raise ActionFileError(
                    f"clip {i}: 'frames' must be a positive integer, got {f!r}."
                )
            clip_frames.append(f)
        if sum(clip_frames) != total:
            raise ActionFileError(
                f"Sum of clip frames ({sum(clip_frames)}) must equal the total frame "
                f"count ({total} = 57 + ({num_iterations} - 1) * 40)."
            )
        return clip_frames

    # --- Mode 2: uniform unit_frames -----------------------------------------
    if unit_frames is not None:
        if not isinstance(unit_frames, int) or isinstance(unit_frames, bool) or unit_frames <= 0:
            raise ActionFileError(
                f"unit_frames must be a positive integer, got {unit_frames!r}."
            )
        n_expected = expected_clip_count(unit_frames, total)
        if len(clips) != n_expected:
            last_frames = total - (n_expected - 1) * unit_frames
            raise ActionFileError(
                f"With unit_frames={unit_frames} and total={total} frames, expected "
                f"{n_expected} clip entries ({n_expected - 1} × {unit_frames} frames + "
                f"1 × {last_frames} frames = {total}), but got {len(clips)} clips.\n"
                f"  Hint: at {VIDEO_FPS} fps, unit_frames={unit_frames} = "
                f"{unit_frames / VIDEO_FPS:.2f} s per entry."
            )
        return unit_clip_frames(unit_frames, total)

    # --- Mode 3: default — one clip per diffusion iteration ------------------
    if len(clips) != num_iterations:
        raise ActionFileError(
            f"Number of clips ({len(clips)}) must equal num_iterations "
            f"({num_iterations}) when neither 'frames' nor 'unit_frames' is set. "
            f"Options:\n"
            f"  a) One clip per iteration (current mode): provide {num_iterations} clips.\n"
            f"  b) Sub-clip granularity: add a top-level \"unit_frames\": N field, "
            f"then provide ceil({total}/N) clips. "
            f"Example: \"unit_frames\": 20 → {expected_clip_count(20, total)} clips "
            f"(0.5 s each at {VIDEO_FPS} fps).\n"
            f"  c) Explicit frames: add a 'frames' field to every clip "
            f"(sum must equal {total})."
        )
    return default_clip_frames(num_iterations)


def _parse_clip_format(data, num_iterations, total, unit_frames=None):
    clips = data["clips"]
    if not isinstance(clips, list) or len(clips) == 0:
        raise ActionFileError("'clips' must be a non-empty list of clip objects.")
    for i, clip in enumerate(clips):
        if not isinstance(clip, dict):
            raise ActionFileError(f"clip {i} must be a JSON object, got {type(clip).__name__}.")

    clip_frames = _resolve_clip_frames(clips, num_iterations, total, unit_frames=unit_frames)

    keyboard_rows = []
    mouse_rows = []
    for i, (clip, n_frames) in enumerate(zip(clips, clip_frames)):
        mouse_tok = str(clip.get("mouse", DEFAULT_MOUSE_TOKEN)).strip().lower()
        kb_tok = str(clip.get("keyboard", DEFAULT_KEYBOARD_TOKEN)).strip().lower()
        if mouse_tok not in MOUSE_TOKEN_MAP:
            raise ActionFileError(
                f"clip {i}: unknown mouse token {mouse_tok!r}. {_supported_tokens_msg()}"
            )
        if kb_tok not in KEYBOARD_TOKEN_MAP:
            raise ActionFileError(
                f"clip {i}: unknown keyboard token {kb_tok!r}. {_supported_tokens_msg()}"
            )
        mouse_vec = MOUSE_TOKEN_MAP[mouse_tok]
        kb_vec = KEYBOARD_TOKEN_MAP[kb_tok]
        for _ in range(n_frames):
            mouse_rows.append(mouse_vec)
            keyboard_rows.append(kb_vec)

    keyboard = torch.tensor(keyboard_rows, dtype=torch.float32)
    mouse = torch.tensor(mouse_rows, dtype=torch.float32)
    if keyboard.shape[0] != total:
        # Should be unreachable given the frame validation above; guard anyway.
        raise ActionFileError(
            f"Expanded clip frames ({keyboard.shape[0]}) do not match expected total "
            f"({total})."
        )
    return keyboard, mouse, clip_frames


def parse_actions_dict(data, num_iterations, unit_frames=None):
    """Parse an already-loaded action dict into (keyboard[T,6], mouse[T,2], clip_frames).

    ``unit_frames`` sets the sub-clip granularity for Schema B when neither
    per-clip ``frames`` fields nor a top-level ``unit_frames`` key are present
    in the dict. The file's ``unit_frames`` key takes precedence over this
    argument.

    Exposed separately so the schema logic can be unit-tested without a file
    on disk.
    """
    total = expected_total_frames(num_iterations)
    if "clips" in data:
        # File-level unit_frames overrides the caller's argument.
        file_uf = data.get("unit_frames", None)
        if file_uf is not None:
            if not isinstance(file_uf, int) or isinstance(file_uf, bool) or file_uf <= 0:
                raise ActionFileError(
                    f"Top-level 'unit_frames' must be a positive integer, got {file_uf!r}."
                )
            unit_frames = file_uf
        return _parse_clip_format(data, num_iterations, total, unit_frames=unit_frames)
    if "keyboard_condition" in data or "mouse_condition" in data:
        return _parse_tensor_format(data, num_iterations, total)
    raise ActionFileError(
        "Unrecognized action schema. Provide either a 'clips' list (clip format) or "
        "both 'keyboard_condition' and 'mouse_condition' arrays (tensor format)."
    )


def load_action_tensors(actions_file, num_iterations, actions_format="json", unit_frames=None):
    """Load and validate an action file.

    ``unit_frames`` is the fallback sub-clip granularity (overridden by a
    top-level ``unit_frames`` key in the file). Set to e.g. 20 for 0.5 s
    control units at 40 fps.

    Returns ``(keyboard_condition [T, 6] float32, mouse_condition [T, 2] float32,
    clip_frames)`` on CPU.
    """
    if actions_format != "json":
        raise ActionFileError(
            f"Unsupported actions_format {actions_format!r}. Only 'json' is supported."
        )
    data = _load_json(actions_file)
    return parse_actions_dict(data, num_iterations, unit_frames=unit_frames)


def build_input_image(pil_image, height, width, device=None, dtype=None):
    """Reproduce the image preprocessing performed by ``utils.utils.get_data``."""
    from .transform import get_video_transform

    input_image = torch.from_numpy(np.array(pil_image)).unsqueeze(0)
    input_image = input_image.permute(0, 3, 1, 2)

    def normalize_to_neg_one_to_one(x):
        return 2.0 * x - 1.0

    transform = get_video_transform(height, width, normalize_to_neg_one_to_one)
    input_image = transform(input_image)
    input_image = input_image.transpose(0, 1).unsqueeze(0)  # b c t h w
    return input_image.to(device, dtype)


def build_extrinsics_from_conditions(keyboard_condition, mouse_condition):
    """Compute extrinsics from frame-level conditions, matching ``get_data``.

    ``keyboard_condition`` is [T, 6] and ``mouse_condition`` is [T, 2] (CPU tensors).
    Returns the extrinsics tensor produced by ``utils.cam_utils.get_extrinsics``.
    """
    from .utils import compute_all_poses_from_actions
    from .cam_utils import get_extrinsics

    first_pose = np.concatenate([np.zeros(3), np.zeros(2)], axis=0)
    all_poses = compute_all_poses_from_actions(
        keyboard_condition, mouse_condition, first_pose=first_pose
    )
    positions = all_poses[:, :3].tolist()
    rotations = np.concatenate(
        [
            np.zeros((all_poses.shape[0], 1)),  # roll = 0
            all_poses[:, 3:5],  # pitch, yaw
        ],
        axis=1,
    ).tolist()
    return get_extrinsics(rotations, positions)


def get_data_from_actions_file(
    actions_file,
    num_iterations,
    height,
    width,
    pil_image,
    device=None,
    dtype=None,
    actions_format="json",
    unit_frames=None,
):
    """Drop-in replacement for ``utils.utils.get_data`` driven by an action file.

    Returns ``(input_image, extrinsics_all, keyboard_condition_all [1, T, 6],
    mouse_condition_all [1, T, 2])``, exactly matching ``get_data``'s contract.
    """
    keyboard_t, mouse_t, _ = load_action_tensors(
        actions_file, num_iterations, actions_format=actions_format,
        unit_frames=unit_frames,
    )
    extrinsics_all = build_extrinsics_from_conditions(keyboard_t, mouse_t)
    input_image = build_input_image(pil_image, height, width, device=device, dtype=dtype)
    return (
        input_image,
        extrinsics_all,
        keyboard_t.to(device, dtype).unsqueeze(0),
        mouse_t.to(device, dtype).unsqueeze(0),
    )


class ReplayActionProvider:
    """Serves precomputed replay actions to the interactive pipeline.

    Holds full-length, frame-level conditions and extrinsics, and slices them
    per diffusion clip using the fixed 57 / 40 schedule (independent of how
    the action file grouped its clips).
    """

    def __init__(self, keyboard_all, mouse_all, extrinsics_full):
        # keyboard_all: [1, T, 6], mouse_all: [1, T, 2], extrinsics_full: [T, ...]
        self.keyboard_all_full = keyboard_all
        self.mouse_all_full = mouse_all
        self.extrinsics_full = extrinsics_full
        self.total_frames = keyboard_all.shape[1]

    def clip_bounds(self, clip_idx):
        """Return (start, end) frame indices for diffusion clip ``clip_idx``."""
        if clip_idx == 0:
            start, frames = 0, FIRST_CLIP_FRAMES
        else:
            start = FIRST_CLIP_FRAMES + (clip_idx - 1) * SUBSEQUENT_CLIP_FRAMES
            frames = SUBSEQUENT_CLIP_FRAMES
        end = start + frames
        if end > self.total_frames:
            raise ActionFileError(
                f"Replay actions exhausted: clip {clip_idx} needs frames "
                f"[{start}, {end}) but only {self.total_frames} frames are available."
            )
        return start, end


def build_replay_provider(
    actions_file, num_iterations, device=None, dtype=None,
    actions_format="json", unit_frames=None,
):
    """Build a :class:`ReplayActionProvider` from an action file."""
    keyboard_t, mouse_t, _ = load_action_tensors(
        actions_file, num_iterations, actions_format=actions_format,
        unit_frames=unit_frames,
    )
    extrinsics_full = build_extrinsics_from_conditions(keyboard_t, mouse_t)
    keyboard_all = keyboard_t.unsqueeze(0).to(device=device, dtype=dtype)
    mouse_all = mouse_t.unsqueeze(0).to(device=device, dtype=dtype)
    return ReplayActionProvider(keyboard_all, mouse_all, extrinsics_full)


def _self_test():
    """Lightweight, dependency-free validation of the schema logic."""
    # --- Existing modes (unchanged) ------------------------------------------

    # Clip format, implicit frames (clip count == num_iterations).
    kb, mo, frames = parse_actions_dict(
        {"clips": [{"mouse": "u", "keyboard": "w"}, {"mouse": "j", "keyboard": "a"}]},
        num_iterations=2,
    )
    assert kb.shape == (97, KEYBOARD_DIM), kb.shape
    assert mo.shape == (97, MOUSE_DIM), mo.shape
    assert frames == [57, 40], frames
    assert kb[0].tolist() == KEYBOARD_TOKEN_MAP["w"]
    assert kb[57].tolist() == KEYBOARD_TOKEN_MAP["a"]
    assert torch.allclose(mo[57], torch.tensor(MOUSE_TOKEN_MAP["j"]))

    # Clip format, explicit per-clip frames.
    kb, mo, frames = parse_actions_dict(
        {
            "clips": [
                {"mouse": "u", "keyboard": "w", "frames": 57},
                {"mouse": "l", "keyboard": "d", "frames": 40},
            ]
        },
        num_iterations=2,
    )
    assert kb.shape[0] == 97 and frames == [57, 40]

    # Tensor format.
    total_1iter = expected_total_frames(1)  # 57
    kb, mo, frames = parse_actions_dict(
        {
            "keyboard_condition": [[0, 0, 1, 0, 0, 0]] * total_1iter,
            "mouse_condition": [[0.0, -0.1]] * total_1iter,
        },
        num_iterations=1,
    )
    assert kb.shape == (57, KEYBOARD_DIM) and mo.shape == (57, MOUSE_DIM)

    # --- unit_frames mode (new) ----------------------------------------------

    # Via top-level JSON field: unit_frames=20, num_iterations=2 → total=97
    # ceil(97/20)=5 clips: [20,20,20,20,17]
    total_2iter = expected_total_frames(2)  # 97
    n_clips_20 = expected_clip_count(20, total_2iter)  # 5
    clips_20 = [{"mouse": "u", "keyboard": "w"}] * n_clips_20
    kb, mo, frames = parse_actions_dict(
        {"unit_frames": 20, "clips": clips_20},
        num_iterations=2,
    )
    assert kb.shape[0] == total_2iter, kb.shape
    assert frames == [20, 20, 20, 20, 17], frames
    # Action at frame 0 and frame 20 both "w"/"u" (same token throughout).
    assert kb[0].tolist() == KEYBOARD_TOKEN_MAP["w"]
    assert kb[96].tolist() == KEYBOARD_TOKEN_MAP["w"]

    # Via caller argument (no file field): same result.
    clips_plain = [{"mouse": "j", "keyboard": "a"}] * n_clips_20
    kb2, mo2, frames2 = parse_actions_dict(
        {"clips": clips_plain},
        num_iterations=2,
        unit_frames=20,
    )
    assert kb2.shape[0] == total_2iter and frames2 == [20, 20, 20, 20, 17]

    # File field overrides caller arg.
    kb3, mo3, frames3 = parse_actions_dict(
        {"unit_frames": 20, "clips": clips_20},
        num_iterations=2,
        unit_frames=99,  # should be ignored
    )
    assert frames3 == [20, 20, 20, 20, 17]

    # unit_frames=40: ceil(97/40)=3 clips → [40, 40, 17]
    n_clips_40 = expected_clip_count(40, total_2iter)  # 3
    clips_40 = [{"mouse": "i", "keyboard": "d"}] * n_clips_40
    kb, mo, frames = parse_actions_dict(
        {"unit_frames": 40, "clips": clips_40},
        num_iterations=2,
    )
    assert frames == [40, 40, 17], frames
    assert kb.shape[0] == total_2iter

    # --- Expected failures ---------------------------------------------------
    def expect_error(fn, needle):
        try:
            fn()
        except ActionFileError as exc:
            assert needle in str(exc), (needle, str(exc))
        else:
            raise AssertionError(f"Expected ActionFileError containing {needle!r}")

    # Wrong clip count without unit_frames.
    expect_error(
        lambda: parse_actions_dict(
            {"clips": [{"mouse": "u", "keyboard": "w"}]}, num_iterations=2
        ),
        "must equal num_iterations",
    )
    # Wrong clip count with unit_frames (via file field).
    expect_error(
        lambda: parse_actions_dict(
            {"unit_frames": 20, "clips": [{"mouse": "u", "keyboard": "w"}] * 3},
            num_iterations=2,
        ),
        "expected 5 clip entries",
    )
    # Wrong clip count with unit_frames (via caller arg).
    expect_error(
        lambda: parse_actions_dict(
            {"clips": [{"mouse": "u", "keyboard": "w"}] * 3},
            num_iterations=2,
            unit_frames=20,
        ),
        "expected 5 clip entries",
    )
    # Cannot mix unit_frames and per-clip frames.
    expect_error(
        lambda: parse_actions_dict(
            {
                "unit_frames": 20,
                "clips": [{"mouse": "u", "keyboard": "w", "frames": 20}] * n_clips_20,
            },
            num_iterations=2,
        ),
        "Cannot mix",
    )
    # Unknown tokens.
    expect_error(
        lambda: parse_actions_dict(
            {"clips": [{"mouse": "z", "keyboard": "w"}]}, num_iterations=1
        ),
        "unknown mouse token",
    )
    expect_error(
        lambda: parse_actions_dict(
            {"clips": [{"mouse": "u", "keyboard": "x"}]}, num_iterations=1
        ),
        "unknown keyboard token",
    )
    # Explicit frames don't sum to total.
    expect_error(
        lambda: parse_actions_dict(
            {
                "clips": [
                    {"mouse": "u", "keyboard": "w", "frames": 10},
                    {"mouse": "u", "keyboard": "w", "frames": 10},
                ]
            },
            num_iterations=2,
        ),
        "must equal the total frame count",
    )
    # Tensor format: wrong total T.
    expect_error(
        lambda: parse_actions_dict(
            {"keyboard_condition": [[0, 0, 1, 0, 0, 0]] * 10, "mouse_condition": [[0.0, 0.0]] * 10},
            num_iterations=1,
        ),
        "Frame-count mismatch",
    )
    # Tensor format: wrong keyboard width.
    expect_error(
        lambda: parse_actions_dict(
            {"keyboard_condition": [[0, 0, 1, 0]] * 57, "mouse_condition": [[0.0, 0.0]] * 57},
            num_iterations=1,
        ),
        "keyboard_condition must have shape",
    )
    # Unrecognized schema.
    expect_error(
        lambda: parse_actions_dict({"foo": 1}, num_iterations=1),
        "Unrecognized action schema",
    )
    print("action_io self-test passed.")


if __name__ == "__main__":
    _self_test()
