"""Robot motion tools. Load legacy convenience exports only when requested.

Configuration inspection and the PartiGen networks do not require CLIP, MuJoCo
or the server's isaac_utils installation.
"""
from importlib import import_module


def __getattr__(name):
    modules = ("dataloader", "skeleton", "model", "diffusion", "dtype", "train", "eval")
    if name in modules:
        return import_module(f".{name}", __name__)
    legacy = {"AutoMldVae": "model.mld_vae", "DenoiserTransformer": "model.mld_denoiser",
              "DenoiserMLP": "model.mld_denoiser"}
    if name in legacy:
        return getattr(import_module(f".{legacy[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
