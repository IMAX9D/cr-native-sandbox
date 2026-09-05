# Standalone offline CR Policy V1

See [Linux setup, data preparation, training and DDP instructions (中文)](README.zh-CN.md).

Install **this directory**, not the parent runtime project:

```bash
python -m pip install -e ./policy_v1
cr-policy-smoke
```

Requires Python 3.8+, NumPy and PyTorch 2.0+. No APK, game assets, simulator or
native runtime is needed for BC. The compiled dataset is supplied separately.
This is an experimental policy, not a claim of measured gameplay strength.
