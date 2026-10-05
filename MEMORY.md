# openpi repository learning memory

## Goal

Understand the repository by tracing one observation through inference and one batch through training.

## Recommended anchor

Use the `pi05_libero` configuration in `src/openpi/training/config.py` because it connects the model, dataset, transforms, training loop, and inference path.

## Reading order

1. Read `README.md`, especially the inference and fine-tuning sections.
2. Read `src/openpi/training/config.py` around `pi05_libero`.
3. Trace inference:
   - `src/openpi/policies/libero_policy.py`
   - `src/openpi/policies/policy_config.py:create_trained_policy`
   - `src/openpi/policies/policy.py:Policy.infer`
   - `src/openpi/models/model.py:Observation`
4. Read `src/openpi/transforms.py` together with `LeRobotLiberoDataConfig` in `config.py`.
5. Read `src/openpi/models/pi0_config.py` and `src/openpi/models/pi0.py`, focusing on `embed_prefix`, `embed_suffix`, `compute_loss`, and `sample_actions`.
6. Trace training through `src/openpi/training/data_loader.py` and `scripts/train.py:train_step`.

## Concepts to track

- Observation keys, shapes, dtypes, and units at every boundary.
- Dataset repacking versus transforms shared by training and inference.
- Image layout, prompt tokenization, normalization, and delta/absolute action conversion.
- Model actions are internally shaped `(batch, action_horizon, action_dim)`; LIBERO uses a 10-step horizon and returns the first 7 action dimensions.
- π₀.₅ changes model behavior through configuration switches while using the π₀ implementation.

## Hands-on checks

```bash
JAX_PLATFORMS=cpu uv run pytest -q src/openpi/transforms_test.py -k 'repack or delta_actions or absolute_actions'
JAX_PLATFORMS=cpu uv run pytest -q src/openpi/training/data_loader_test.py::test_with_fake_dataset
```

Then inspect `scripts/train_test.py`, which runs a short dummy training job and tests checkpoint resuming.

## Milestone

Be able to explain why a LIBERO observation becomes a 10-step, 7-dimensional action chunk, identify every transform involved, and point to where training calls the model loss.
