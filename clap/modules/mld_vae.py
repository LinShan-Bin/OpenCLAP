from functools import reduce
from typing import List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions.distribution import Distribution

from .position_encoding_layer import PositionalEncoding
from .cross_attention import (
    SkipTransformerEncoder,
    SkipTransformerDecoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from .position_encoding import build_position_encoding

"""
vae

skip connection encoder 
skip connection decoder

mem for each decoder layer
"""

def lengths_to_mask(lengths: List[int],
                    device: torch.device,
                    max_len: int = None) -> Tensor:
    lengths = torch.tensor(lengths, device=device)
    max_len = max_len if max_len else max(lengths)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


class MldVae(nn.Module):

    def __init__(self,
                 ablation,
                 nfeats: int,
                 latent_dim: list = [1, 256],
                 codebook_dim: int = -1,
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 arch: str = "all_encoder",
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 position_embedding: str = "learned",
                 **kwargs) -> None:

        super().__init__()

        self.latent_size = latent_dim[0]
        self.latent_dim = latent_dim[-1]
        self.codebook_dim = codebook_dim
        input_feats = nfeats
        output_feats = nfeats
        self.arch = arch
        self.mlp_dist = ablation.MLP_DIST
        self.pe_type = ablation.PE_TYPE

        if self.pe_type == "actor":
            self.query_pos_encoder = PositionalEncoding(
                self.latent_dim, dropout)
            self.query_pos_decoder = PositionalEncoding(
                self.latent_dim, dropout)
        elif self.pe_type == "mld":
            self.query_pos_encoder = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)
            self.query_pos_decoder = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)
        else:
            raise ValueError("Not Support PE type")

        encoder_layer = TransformerEncoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        encoder_norm = nn.LayerNorm(self.latent_dim)
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers,
                                              encoder_norm)

        if self.arch == "all_encoder":
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = SkipTransformerEncoder(encoder_layer, num_layers,
                                                  decoder_norm)
        elif self.arch == "encoder_decoder":
            decoder_layer = TransformerDecoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = SkipTransformerDecoder(decoder_layer, num_layers,
                                                  decoder_norm)
        else:
            raise ValueError("Not support architecture!")

        if self.mlp_dist:
            self.global_motion_token = nn.Parameter(
                torch.randn(self.latent_size, self.latent_dim))
            self.dist_layer = nn.Linear(self.latent_dim, self.latent_dim)  # Only output mu, no logvar
        else:
            self.global_motion_token = nn.Parameter(
                torch.randn(self.latent_size, self.latent_dim))  # Only need latent_size, not *2

        self.skel_embedding = nn.Linear(input_feats, self.latent_dim)
        self.final_layer = nn.Linear(self.latent_dim, output_feats)

        # Optional projections for codebook space
        if self.codebook_dim != self.latent_dim:
            self.to_codebook = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.GELU(),
                nn.Linear(self.latent_dim, self.codebook_dim),
            )
            self.from_codebook = nn.Sequential(
                nn.Linear(self.codebook_dim, self.latent_dim),
                nn.GELU(),
                nn.Linear(self.latent_dim, self.latent_dim),
            )
        else:
            self.to_codebook = nn.Identity()
            self.from_codebook = nn.Identity()

    def forward(self, features: Tensor, lengths: Optional[List[int]] = None):
        # Temp
        # Todo
        # remove and test this function
        print("Should Not enter here")

        z, _ = self.encode(features, lengths)  # dist not needed anymore
        feats_rst = self.decode(z, lengths)
        return feats_rst, z, None  # No distribution returned

    def encode(
            self,
            features: Tensor,
            lengths: Optional[List[int]] = None
    ) -> Tuple[Tensor, None]:
        if lengths is None:
            lengths = [len(feature) for feature in features]

        device = features.device

        bs, nframes, nfeats = features.shape
        mask = lengths_to_mask(lengths, device)

        x = features
        # Embed each human poses into latent vectors
        x = self.skel_embedding(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        # Each batch has its own set of tokens
        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))

        # create a bigger mask, to allow attend to emb
        dist_masks = torch.ones((bs, dist.shape[0]),
                                dtype=bool,
                                device=x.device)
        aug_mask = torch.cat((dist_masks, mask), 1)

        # adding the embedding token for all sequences
        xseq = torch.cat((dist, x), 0)

        if self.pe_type == "actor":
            xseq = self.query_pos_encoder(xseq)
            dist = self.encoder(xseq,
                                src_key_padding_mask=~aug_mask)[:dist.shape[0]]
        elif self.pe_type == "mld":
            xseq = self.query_pos_encoder(xseq)
            dist = self.encoder(xseq,
                                src_key_padding_mask=~aug_mask)[:dist.shape[0]]
            # query_pos = self.query_pos_encoder(xseq)
            # dist = self.encoder(xseq, pos=query_pos, src_key_padding_mask=~aug_mask)[
            #     : dist.shape[0]
            # ]

        # content distribution - only mu, no logvar needed
        if self.mlp_dist:
            mu = self.dist_layer(dist)  # Direct output, no need to slice
        else:
            mu = dist  # Use all tokens since we only have latent_size now

        # Return feature directly without sampling
        latent = mu  # Directly use mean as feature
        latent = self.to_codebook(latent)
        return latent, None  # No distribution needed

    def decode(self, z: Tensor, lengths: List[int]):
        mask = lengths_to_mask(lengths, z.device)
        bs, nframes = mask.shape

        queries = torch.zeros(nframes, bs, self.latent_dim, device=z.device)

        # Project codebook latent back to transformer latent dim
        z = self.from_codebook(z)

        # todo
        # investigate the motion middle error!!!

        # Pass through the transformer decoder
        # with the latent vector for memory
        if self.arch == "all_encoder":
            xseq = torch.cat((z, queries), axis=0)
            z_mask = torch.ones((bs, z.shape[0]),
                                dtype=bool,
                                device=z.device)
            augmask = torch.cat((z_mask, mask), axis=1)

            if self.pe_type == "actor":
                xseq = self.query_pos_decoder(xseq)
                output = self.decoder(
                    xseq, src_key_padding_mask=~augmask)[z.shape[0]:]
            elif self.pe_type == "mld":
                xseq = self.query_pos_decoder(xseq)
                output = self.decoder(
                    xseq, src_key_padding_mask=~augmask)[z.shape[0]:]
                # query_pos = self.query_pos_decoder(xseq)
                # output = self.decoder(
                #     xseq, pos=query_pos, src_key_padding_mask=~augmask
                # )[z.shape[0] :]

        elif self.arch == "encoder_decoder":
            if self.pe_type == "actor":
                queries = self.query_pos_decoder(queries)
                output = self.decoder(tgt=queries,
                                      memory=z,
                                      tgt_key_padding_mask=~mask).squeeze(0)
            elif self.pe_type == "mld":
                queries = self.query_pos_decoder(queries)
                # mem_pos = self.mem_pos_decoder(z)
                output = self.decoder(
                    tgt=queries,
                    memory=z,
                    tgt_key_padding_mask=~mask,
                    # query_pos=query_pos,
                    # pos=mem_pos,
                ).squeeze(0)
                # query_pos = self.query_pos_decoder(queries)
                # # mem_pos = self.mem_pos_decoder(z)
                # output = self.decoder(
                #     tgt=queries,
                #     memory=z,
                #     tgt_key_padding_mask=~mask,
                #     query_pos=query_pos,
                #     # pos=mem_pos,
                # ).squeeze(0)

        output = self.final_layer(output)
        # zero for padded area
        output[~mask.T] = 0
        # Pytorch Transformer: [Sequence, Batch size, ...]
        feats = output.permute(1, 0, 2)
        return feats
