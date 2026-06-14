# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import argparse
import logging
import os
import socket

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


def _seed_everything(seed: int) -> None:
    """Seed Python / NumPy / PyTorch (CPU + CUDA) for reproducible inference."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info("Seeded inference RNGs to %d", seed)


def main(args) -> None:
    """Build the policy wrapper and start the websocket server.

    The wrapper now owns un-normalization + chunk_size discovery so that all
    eval clients (LIBERO / SimplerEnv / etc.) just need to forward `examples`
    and consume already-unnormalized actions from the response.
    """
    if args.seed is not None:
        _seed_everything(args.seed)

    wrapper = PolicyServerWrapper(
        ckpt_path=args.ckpt_path,
        device="cuda",
        use_bf16=args.use_bf16,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # start websocket server; wrapper.metadata is sent at handshake.
    server = WebsocketPolicyServer(
        policy=wrapper,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=wrapper.metadata,
    )
    logging.info("server running ... metadata=%s", wrapper.metadata)
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--idle_timeout", type=int, default=1800, help="Idle timeout in seconds, -1 means never close")
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="If set, seed Python/NumPy/PyTorch (incl. CUDA) at server startup so that the "
             "flow-matching action sampling is reproducible across server restarts.",
    )
    return parser


def start_debugpy_once():
    """start debugpy once"""
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10095))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10095 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    if os.getenv("DEBUG", False):
        print("🔍 DEBUGPY is enabled")
        start_debugpy_once()
    main(args)
