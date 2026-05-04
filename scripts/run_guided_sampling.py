"""Guidance-loop driver: pilot run first, then full-scale.

Two-phase workflow to avoid wasting GPU hours on a guidance setup that
goes the wrong way:

    Phase 1 — Pilot (default):
        Sample N=8 plans (configurable). For each plan, compute the
        unguided final compliance energy and the guided final compliance
        energy under the requested jurisdiction. Print a clear go/no-go
        verdict:

            GO   if  median(unguided_energy - guided_energy) > 0
                 AND fraction_with_improvement >= 0.6
            NO-GO otherwise.

        Pilot finishes in <1 min on the MockDiffusionBackbone (CPU) and
        in roughly N * single-step-time on a real GPU backbone.

    Phase 2 — Full run (`--full`):
        Only entered after a successful pilot. Sample the requested
        number of plans, dump trajectories.

Usage examples
--------------

    # Pilot against the mock backbone (CPU, instant — sanity check):
    python scripts/run_guided_sampling.py --jurisdiction TR

    # Pilot against a real HouseDiffusion checkpoint (forked venv):
    python scripts/run_guided_sampling.py \
        --jurisdiction TR \
        --backbone housediffusion \
        --checkpoint data/HouseDiffusion/checkpoints/msd_ema.pt

    # Full run (only after pilot says GO):
    python scripts/run_guided_sampling.py \
        --jurisdiction TR --backbone housediffusion \
        --checkpoint data/HouseDiffusion/checkpoints/msd_ema.pt \
        --full --n-plans 64

The driver writes trajectories and per-run metrics to
``output/guided_sampling/<run-id>/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from code_module import (  # noqa: E402
    JURISDICTIONS,
    door_width_energy,
    corridor_width_energy,
    egress_distance_energy,
)
from code_module.guided_sampling import universal_guidance_sample  # noqa: E402
from model_module import MockDiffusionBackbone  # noqa: E402

logger = logging.getLogger("guided_sampling")


# ---------------------------------------------------------------------------
# Energy assembly
# ---------------------------------------------------------------------------

# Stage-3 first-batch rules (see Knowledge/stage3-rule-selection.md).
# These three are the ones with implemented differentiable energies.
SOFT_GUIDANCE_RULES = ("door_width", "corridor_min_width", "egress_travel_distance")


def jurisdiction_energy_fn(
    code: str, *, weight_per_rule: float = 1.0
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a scalar energy function from a jurisdiction's soft-rule thresholds.

    The energy is a tensor-only sum so it can be applied to whatever
    shape ``predict_x0`` returns. Concrete coordinate semantics are the
    backbone's responsibility — for the mock backbone, we treat the
    tensor as a stack of door corners shape (B, 4, 2).

    HouseDiffusion produces (B, N_rooms, 32, 2). A shape adapter for that
    case is a separate Stage-3 sub-task; until it exists, the corridor and
    egress terms silently return zero on (B, 32, 2) inputs. The driver's
    ``--validate-shape`` flag surfaces this before any GPU work.
    """
    spec_rules = dict(JURISDICTIONS[code].rules)
    door_kwargs = spec_rules.get("door_width", {"min_width_m": 0.85})
    corridor_kwargs = spec_rules.get("corridor_min_width", {"min_width_m": 1.20})
    egress_kwargs = spec_rules.get(
        "egress_travel_distance", {"max_distance_m": 30.0}
    )

    min_door = float(door_kwargs.get("min_width_m", 0.85))
    min_corr = float(corridor_kwargs.get("min_width_m", 1.20))
    max_egress = float(egress_kwargs.get("max_distance_m", 30.0))

    def energy(x: torch.Tensor) -> torch.Tensor:
        # Each term is tensor-only and broadcasts across the leading axes.
        e_door = door_width_energy(x, min_width_m=min_door)
        # corridor_width_energy expects a (B, N, 4, 2) layout in the same
        # convention as door_width_energy; if the backbone produces a
        # different shape the user can pass --soft-rules door_width to
        # exercise just the door term while the others are wired up.
        try:
            e_corr = corridor_width_energy(x, min_width_m=min_corr)
        except Exception:  # noqa: BLE001
            e_corr = torch.zeros((), dtype=x.dtype, device=x.device)
        try:
            e_egress = egress_distance_energy(x, max_distance_m=max_egress)
        except Exception:  # noqa: BLE001
            e_egress = torch.zeros((), dtype=x.dtype, device=x.device)
        return weight_per_rule * (e_door + e_corr + e_egress)

    return energy


# ---------------------------------------------------------------------------
# Pilot-vs-Full workflow
# ---------------------------------------------------------------------------


@dataclass
class PerPlanResult:
    plan_idx: int
    unguided_final_energy: float
    guided_final_energy: float
    delta_energy: float  # unguided - guided; positive means guidance helped
    guided_grad_norm_max: float

    @property
    def improved(self) -> bool:
        return self.delta_energy > 0


@dataclass
class PilotVerdict:
    n_plans: int
    median_delta: float
    fraction_improved: float
    go: bool
    reason: str


def evaluate_pilot(results: list[PerPlanResult]) -> PilotVerdict:
    """Return GO/NO-GO based on guided vs unguided energy distribution."""
    deltas = [r.delta_energy for r in results]
    improved = [r.improved for r in results]
    median_delta = statistics.median(deltas) if deltas else 0.0
    fraction = sum(improved) / max(len(improved), 1)

    go = median_delta > 0 and fraction >= 0.6
    if go:
        reason = (
            f"median_delta={median_delta:.4f} > 0 and "
            f"fraction_improved={fraction:.2f} >= 0.60"
        )
    else:
        reason = (
            f"median_delta={median_delta:.4f}, "
            f"fraction_improved={fraction:.2f}; "
            "guidance is not consistently reducing energy — check sign of "
            "energy fn, guidance_scale, or jurisdiction wiring"
        )
    return PilotVerdict(
        n_plans=len(results),
        median_delta=median_delta,
        fraction_improved=fraction,
        go=go,
        reason=reason,
    )


def run_one_plan(
    backbone,
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    plan_idx: int,
    guidance_scale: float,
    seed: int,
    sample_shape: tuple[int, ...],
) -> PerPlanResult:
    g = torch.Generator().manual_seed(seed + plan_idx)
    x_T = torch.randn(*sample_shape, generator=g, dtype=torch.float64)

    _, h_unguided = universal_guidance_sample(
        backbone, x_T.clone(), energy_fn, guidance_scale=0.0
    )
    _, h_guided = universal_guidance_sample(
        backbone, x_T.clone(), energy_fn, guidance_scale=guidance_scale
    )

    final_unguided = h_unguided.energies[-1] if h_unguided.energies else 0.0
    final_guided = h_guided.energies[-1] if h_guided.energies else 0.0
    grad_max = max(h_guided.grad_norms) if h_guided.grad_norms else 0.0
    return PerPlanResult(
        plan_idx=plan_idx,
        unguided_final_energy=final_unguided,
        guided_final_energy=final_guided,
        delta_energy=final_unguided - final_guided,
        guided_grad_norm_max=grad_max,
    )


def run_phase(
    backbone,
    energy_fn,
    *,
    n_plans: int,
    guidance_scale: float,
    seed: int,
    sample_shape: tuple[int, ...],
    label: str,
) -> list[PerPlanResult]:
    logger.info("Phase '%s': %d plans, guidance_scale=%.3f", label, n_plans, guidance_scale)
    results: list[PerPlanResult] = []
    t0 = time.time()
    for i in range(n_plans):
        r = run_one_plan(
            backbone,
            energy_fn,
            plan_idx=i,
            guidance_scale=guidance_scale,
            seed=seed,
            sample_shape=sample_shape,
        )
        results.append(r)
        logger.info(
            "  plan %d/%d  delta=%+.4f  grad_max=%.4f  improved=%s",
            i + 1, n_plans, r.delta_energy, r.guided_grad_norm_max, r.improved,
        )
    elapsed = time.time() - t0
    logger.info("Phase '%s' finished in %.1fs", label, elapsed)
    return results


# ---------------------------------------------------------------------------
# Backbone factories
# ---------------------------------------------------------------------------


def preflight_validate_shape(
    jurisdiction_code: str, sample_shape: tuple[int, ...]
) -> bool:
    """Validate that the energy fn semantically matches the backbone shape.

    Two checks per soft-rule term:
        1. Negative control — a clearly-violating input (5 cm coords) must
           produce energy > 0.
        2. Positive control — a compliant input (a 1 m square per polygon,
           rotated-rect-ordered) must produce energy ≈ 0 for door/corridor
           terms.

    Both must hold for the term to be considered correctly wired. If only
    the negative control passes, the energy fn is producing plausible-
    looking numbers but does not actually encode the rule on this shape
    (the silent-failure case). The mock backbone uses (B, 4, 2) which the
    rule energies were written against; HouseDiffusion uses (B, N_rooms,
    32, 2) which needs a shape adapter.
    """
    spec_rules = dict(JURISDICTIONS[jurisdiction_code].rules)
    min_door = float(spec_rules.get("door_width", {}).get("min_width_m", 0.85))
    min_corr = float(
        spec_rules.get("corridor_min_width", {}).get("min_width_m", 1.20)
    )

    # The rule energies (door_width_energy, corridor_width_energy) expect a
    # tensor of shape (..., 4, 2) — 4 rotated-rect corners per polygon.
    # Reject any sample_shape whose second-to-last dim is not 4.
    if len(sample_shape) < 2 or sample_shape[-2] != 4:
        logger.error(
            "Pre-flight: sample_shape %s does not match the (..., 4, 2) "
            "rotated-rect layout the soft-rule energies were written for. "
            "HouseDiffusion uses (B, N_rooms, 32, 2); add a shape adapter "
            "to jurisdiction_energy_fn that converts 32-vertex polygons to "
            "rotated rect corners before the energy fn is called.",
            sample_shape,
        )
        return False

    torch.manual_seed(0)

    # --- Negative control: clearly-violating ---
    x_bad = torch.rand(*sample_shape, dtype=torch.float64) * 0.05  # ~5 cm

    # --- Positive control: 1 m unit square at the origin, rotated-rect order ---
    # Tile the four corners across whatever leading dims the backbone uses.
    unit = torch.tensor(
        [[0.0, 0.0], [1.5, 0.0], [1.5, 1.5], [0.0, 1.5]], dtype=torch.float64
    )
    x_good = unit.expand(*sample_shape).contiguous()

    def _energies(x: torch.Tensor) -> dict[str, float]:
        return {
            "door_width": float(door_width_energy(x, min_width_m=min_door)),
            "corridor_min_width": float(
                corridor_width_energy(x, min_width_m=min_corr)
            ),
        }

    bad = _energies(x_bad)
    good = _energies(x_good)

    logger.info("Pre-flight shape validation on %s:", sample_shape)
    logger.info("  rule                    violating    compliant   verdict")
    all_ok = True
    for name in ("door_width", "corridor_min_width"):
        violating_positive = bad[name] > 1e-6
        compliant_zero = good[name] < 1e-6
        ok = violating_positive and compliant_zero
        all_ok &= ok
        verdict = "OK" if ok else (
            "WIRED_WRONG"
            if not compliant_zero
            else "INACTIVE"
        )
        logger.info(
            "  %-22s %10.4f   %10.6f   %s",
            name, bad[name], good[name], verdict,
        )

    if not all_ok:
        logger.error(
            "Pre-flight FAILED — at least one rule energy does not match "
            "this backbone's tensor shape. Fix the energy fn or add a "
            "decode adapter before running pilot/full."
        )
    return all_ok


def make_backbone(name: str, *, checkpoint: Optional[str] = None) -> tuple:
    """Return (backbone, sample_shape).

    sample_shape is the per-sample tensor shape, *without* batch dimension.
    """
    if name == "mock":
        target = torch.tensor(
            [[[0.0, 0.0], [0.70, 0.0], [0.70, 0.10], [0.0, 0.10]]],
            dtype=torch.float64,
        )
        return MockDiffusionBackbone(target=target, n_steps=30), tuple(target.shape)
    if name == "housediffusion":
        from model_module import load_housediffusion_backbone

        if checkpoint is None:
            raise SystemExit(
                "housediffusion backbone requires --checkpoint <path> "
                "(activate the forked venv first: source data/HouseDiffusion/.venv-hd/bin/activate)"
            )
        backbone = load_housediffusion_backbone(checkpoint)
        # HouseDiffusion samples (B, N=32, 2) corner tensors. The driver
        # currently runs B=1; the energy_fn handles arbitrary leading dims.
        return backbone, (1, 32, 2)
    raise SystemExit(f"unknown --backbone '{name}' (choose: mock, housediffusion)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--jurisdiction", required=True, choices=sorted(JURISDICTIONS))
    p.add_argument(
        "--backbone",
        default="mock",
        choices=("mock", "housediffusion"),
        help="diffusion backbone to wrap (default: mock)",
    )
    p.add_argument("--checkpoint", help="checkpoint path for real backbones")
    p.add_argument(
        "--n-plans",
        type=int,
        default=8,
        help="pilot defaults to 8; raise for full-scale runs",
    )
    p.add_argument(
        "--guidance-scale",
        type=float,
        default=0.5,
        help="universal-guidance step size",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--full",
        action="store_true",
        help=(
            "Skip pilot gating and run --n-plans plans directly. Only use "
            "after a previous pilot returned GO."
        ),
    )
    p.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "output" / "guided_sampling"),
        help="root for run artifacts",
    )
    p.add_argument(
        "--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING")
    )
    p.add_argument(
        "--validate-shape",
        action="store_true",
        help=(
            "Pre-flight: build a random sample of sample_shape, evaluate the "
            "energy fn, and assert each soft-rule term is non-zero. Catches "
            "shape mismatches between the backbone output and the energy fn "
            "before GPU spend."
        ),
    )
    args = p.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    backbone, sample_shape = make_backbone(args.backbone, checkpoint=args.checkpoint)
    energy_fn = jurisdiction_energy_fn(args.jurisdiction)

    if args.validate_shape:
        ok = preflight_validate_shape(args.jurisdiction, sample_shape)
        if not ok:
            return 2

    run_id = f"{args.jurisdiction}-{args.backbone}-{int(time.time())}"
    out_dir = Path(args.out_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pilot_n = min(args.n_plans, 8) if not args.full else 0

    if pilot_n > 0:
        logger.info("=" * 60)
        logger.info("PHASE 1 — pilot (%d plans, gating decision)", pilot_n)
        logger.info("=" * 60)
        pilot = run_phase(
            backbone,
            energy_fn,
            n_plans=pilot_n,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            sample_shape=sample_shape,
            label="pilot",
        )
        verdict = evaluate_pilot(pilot)
        (out_dir / "pilot.json").write_text(
            json.dumps(
                {
                    "verdict": asdict(verdict),
                    "results": [asdict(r) for r in pilot],
                    "args": vars(args),
                },
                indent=2,
            )
        )
        logger.info("PILOT VERDICT: %s", "GO" if verdict.go else "NO-GO")
        logger.info("  reason: %s", verdict.reason)
        if not verdict.go:
            logger.warning(
                "Pilot did NOT pass — refusing to enter full-scale phase. "
                "Re-run with adjusted --guidance-scale or fix the energy fn, "
                "then retry. Pass --full to bypass gating (not recommended)."
            )
            return 1
        if args.n_plans <= pilot_n:
            # User asked for a small run that the pilot already covered.
            return 0

    logger.info("=" * 60)
    logger.info("PHASE 2 — full run (%d plans)", args.n_plans)
    logger.info("=" * 60)
    full = run_phase(
        backbone,
        energy_fn,
        n_plans=args.n_plans,
        guidance_scale=args.guidance_scale,
        seed=args.seed + 1000,  # different seed offset from pilot
        sample_shape=sample_shape,
        label="full",
    )
    verdict_full = evaluate_pilot(full)  # same metric, just for reporting
    (out_dir / "full.json").write_text(
        json.dumps(
            {
                "verdict": asdict(verdict_full),
                "results": [asdict(r) for r in full],
                "args": vars(args),
            },
            indent=2,
        )
    )
    logger.info("Full run summary:")
    logger.info("  median_delta:      %+.4f", verdict_full.median_delta)
    logger.info("  fraction_improved: %.2f", verdict_full.fraction_improved)
    logger.info("Artifacts written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
