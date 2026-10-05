"""Run a bounded CPU smoke test of the π₀ training pipeline on synthetic data.

Usage: uv run scripts/train_tiny.py
"""

import argparse
import dataclasses
import os
import tempfile

# Set these before importing JAX through openpi.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1, help="Training steps (default: 1)")
    parser.add_argument("--samples", type=int, default=16, help="Synthetic examples, 1-99 (default: 16)")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if not 1 <= args.samples < 100:
        parser.error("--samples must be between 1 and 99")

    import train

    from openpi.models.pi0_config import Pi0Config
    from openpi.training import config as _config

    model = Pi0Config(
        dtype="float32",
        paligemma_variant="tiny",
        action_expert_variant="tiny",
        vision_variant="nano/56",
        action_dim=8,
        action_horizon=4,
        max_token_len=8,
    )
    with tempfile.TemporaryDirectory(prefix="openpi-tiny-") as checkpoint_dir:
        config = dataclasses.replace(
            _config.get_config("debug"),
            model=model,
            data=_config.FakeDataConfig(num_samples=args.samples),
            checkpoint_base_dir=checkpoint_dir,
            exp_name="smoke",
            batch_size=1,
            num_workers=0,
            num_train_steps=args.steps,
            log_interval=1,
            save_interval=args.steps,
            keep_period=None,
            ema_decay=None,
            overwrite=False,
            wandb_enabled=False,
        )
        print(f"Training for {args.steps} step(s) on {args.samples} synthetic examples (CPU, hidden width 8).")
        train.main(config)
    print("Tiny training run completed.")


if __name__ == "__main__":
    main()
