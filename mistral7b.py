from typing import List, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import torch
from mistral_inference.cache import (
    BufferCache,
)
from mistral_inference.transformer import Transformer


@dataclass
class SimpleInputMetadata:
    # rope absolute positions
    positions: torch.Tensor

    @staticmethod
    def from_seqlens(seqlens: List[int], device: torch.device) -> "SimpleInputMetadata":
        return SimpleInputMetadata(
            positions=torch.cat([torch.arange(0, seqlen) for seqlen in seqlens]).to(
                device=device, dtype=torch.long
            )
        )
    

class SteerableTransformer(Transformer):
    def __init__(self, *args, **kwargs):
        # Call the parent class's initialization
        super(SteerableTransformer, self).__init__(*args, **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        seqlens: List[int],
        cache: Optional[BufferCache] = None,
        using_sae: bool = False,
        sae=None,
        features=None,
    ) -> torch.Tensor:
        """
        Overrides the forward method to optionally use SAE (Sparse Autoencoder).
        If `using_sae` is True and SAE is provided, it replaces activations with SAE reconstructions.
        """
        # If using SAE, apply the custom forward logic with SAE
        if using_sae and sae is not None:
            h = self._forward_with_sae(input_ids, seqlens, cache, sae, features)
        else:
            # Otherwise, call the base class's forward method (which does not expect sae arguments)
            h = super(SteerableTransformer, self).forward_partial(
                input_ids, seqlens, cache=cache
            )

        # Handle the output, same as in the original Transformer forward
        if self.pipeline_rank < self.num_pipeline_ranks - 1:
            outs = torch.empty(
                h.shape[0], self.vocab_size, device=h.device, dtype=h.dtype
            )
        else:
            assert self.output is not None
            outs = self.output(h)

        # Broadcast output if multiple pipeline ranks are used
        if self.num_pipeline_ranks > 1:
            torch.distributed.broadcast(outs, src=self.num_pipeline_ranks - 1)

        return outs.float()

    def _forward_with_sae(
        self,
        input_ids: torch.Tensor,
        seqlens: List[int],
        cache: Optional[BufferCache],
        sae,
        features
    ) -> torch.Tensor:
        """
        Custom forward method that uses SAE (Sparse Autoencoder) for activations.
        """
        (num_toks,) = input_ids.shape
        input_metadata = SimpleInputMetadata.from_seqlens(seqlens, self.device)

        # Embedding processing for the first pipeline rank
        if self.pipeline_rank == 0:
            assert self.tok_embeddings is not None
            h = self.tok_embeddings(input_ids)
        else:
            h = torch.empty(num_toks, self.args.dim, device=self.device, dtype=self.dtype)
            torch.distributed.recv(h, src=self.pipeline_rank - 1)

        freqs_cis = self.freqs_cis[input_metadata.positions]

        # Process through the layers and apply SAE at the appropriate layer
        for local_layer_id, layer in enumerate(self.layers.values()):
            if cache is not None:
                cache_view = cache.get_view(local_layer_id, input_metadata)
            else:
                cache_view = None

            h = layer(h, freqs_cis, cache_view)

            # Apply SAE at the specified layer (assuming at all layers or specific ones)
            if sae is not None:
                with torch.no_grad():
                    h = sae.forward_val(h, features=features).to(torch.bfloat16)

        if cache is not None:
            cache.update_seqlens(seqlens)

        return h

    @staticmethod
    def from_folder(
        folder: Union[Path, str],
        max_batch_size: int = 32768,
        num_pipeline_ranks: int = 1,
        device: Union[torch.device, str] = "cuda",
        dtype: Optional[torch.dtype] = None,
    ) -> "SteerableTransformer":
        """
        Use the base class's from_folder method to load a Transformer,
        then convert the loaded model to a SteerableTransformer.
        """
        # Use super() to call the original Transformer from_folder
        model = super(SteerableTransformer, SteerableTransformer).from_folder(
            folder, max_batch_size, num_pipeline_ranks, device, dtype
        )
        
        # Convert the returned model to SteerableTransformer
        model.__class__ = SteerableTransformer

        return model