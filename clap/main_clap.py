"""LightningCLI entry-point for CLAP Stage 1 (Act-VAE) and Stage 2 (VD-VAE).

Run::

    python -m clap.main_clap fit  --config clap/configs/clap-s1-l32.yaml
    python -m clap.main_clap fit  --config clap/configs/clap-s2-l32.yaml
    python -m clap.main_clap test --config clap/configs/clap-s2-l32.yaml --ckpt_path .../last.ckpt

The model class is selected by the ``model.stage`` field in the YAML
(``stage-1`` vs ``stage-2``); both share the same :class:`DINO_CLAP`
LightningModule.
"""
from lightning.pytorch.cli import LightningCLI

from clap.dataset_lerobot import LightningLerobot
from clap.model_clap import DINO_CLAP


def cli_main():
    return LightningCLI(
        DINO_CLAP,
        LightningLerobot,
        seed_everything_default=42,
    )


if __name__ == "__main__":
    cli_main()
