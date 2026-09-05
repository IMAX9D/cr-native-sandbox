# Standalone offline CR Policy V1

See [Linux setup, data preparation, training and DDP instructions (中文)](README.zh-CN.md).

Install **this directory**, not the parent runtime project:

```bash
python -m pip install --upgrade pip==24.3.1 setuptools==75.3.2 wheel==0.45.1
python -m pip install --no-build-isolation -e ./policy_v1
cr-policy-smoke
```

Requires Python 3.8+, NumPy and PyTorch 2.0+. No APK, game assets, simulator or
native runtime is needed for BC. The compiled dataset is supplied separately.
This is an experimental policy, not a claim of measured gameplay strength.
