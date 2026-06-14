"""CLAP model modules.

Re-exports the latent-action / vision-dynamic VQ-VAE families used in the
paper (Stage 1 / Stage 2 / Stage 3). The legacy LAM / LAPA modules from the
upstream UniVLA codebase are not bundled here — open them in the upstream
repo if you need them.
"""

from clap.modules.clap import (
    ContrastiveDINOLatentActionModel,
    LatentActionModel,
    DualBranchLatentActionModel,
)
