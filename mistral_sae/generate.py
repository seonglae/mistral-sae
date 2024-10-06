import torch
from typing import List, Optional, Tuple

# Import the existing generate function and necessary classes
from mistral_inference.generate import sample
from mistral_sae.model import SteerableTransformer


@torch.inference_mode()
def generate_with_sae(
    encoded_prompts: List[List[int]],
    model: SteerableTransformer,
    *,
    max_tokens: int,
    temperature: float,
    eos_id: Optional[int] = None,
    sae=None,
    features=None,
    top_p: float = 0.8,
) -> Tuple[List[List[int]], List[List[float]]]:
    model.eval()
    B, _ = len(encoded_prompts), model.args.vocab_size

    # Bookkeeping
    logprobs: List[List[float]] = [[] for _ in range(B)]
    generated_tokens: List[List[int]] = [[] for _ in range(B)]
    is_finished = [False for _ in range(B)]

    max_prompt_len = max(len(p) for p in encoded_prompts)

    # Initialize cache if needed
    cache = None  # Modify if you plan to use caching

    for i in range(max_prompt_len + max_tokens):
        current_tokens = []
        current_seqlens = []

        for b in range(B):
            if i < len(encoded_prompts[b]):
                # Still processing prompt
                seq = encoded_prompts[b][: i + 1]
            elif not is_finished[b]:
                # Generating new tokens
                seq = encoded_prompts[b] + generated_tokens[b]
            else:
                # This sequence is finished
                continue

            current_tokens.extend(seq)
            current_seqlens.append(len(seq))

        if not current_tokens:
            break

        input_tensor = torch.tensor(
            current_tokens, device=model.device, dtype=torch.long
        )

        # Determine if we're processing input tokens or generating
        is_generating = i >= max_prompt_len

        # Forward pass through the model
        prelogits = model.forward(
            input_tensor,
            seqlens=current_seqlens,
            cache=cache,  # No cache in this example
            using_sae=is_generating,
            sae=sae if is_generating else None,
            features=features,
        )

        logits = torch.log_softmax(prelogits, dim=-1)

        # Process logits for each sequence
        offset = 0
        for b in range(B):
            if i < len(encoded_prompts[b]) - 1:
                # Still processing prompt, record logprob
                idx = offset + i
                next_token_id = encoded_prompts[b][i + 1]
                logprobs[b].append(logits[idx, next_token_id].item())
            elif i >= len(encoded_prompts[b]) and not is_finished[b]:
                # Generate next token
                seq_logits = logits[
                    offset + current_seqlens[b] - 1 : offset + current_seqlens[b]
                ]
                next_token = sample(seq_logits, temperature=temperature, top_p=top_p)
                generated_tokens[b].append(next_token.item())
                logprobs[b].append(seq_logits[0, next_token.item()].item())

                if eos_id is not None and next_token.item() == eos_id:
                    is_finished[b] = True

            offset += current_seqlens[b]

        if all(is_finished):
            break

    return generated_tokens, logprobs

@torch.inference_mode()
def get_input_activations_at_layer(
    encoded_prompts: List[List[int]],
    model: SteerableTransformer,
    target_layer: int,
    *,
    chunk_size: Optional[int] = None
) -> torch.Tensor:
    model = model.eval()
    with torch.no_grad():
        seqlens = [len(x) for x in encoded_prompts]

        # One chunk if size not specified
        max_prompt_len = max(seqlens)
        if chunk_size is None:
            chunk_size = max_prompt_len

        # List to store activations for the target layer
        layer_activations = []

        # Encode prompt by chunks and get activations
        for s in range(0, max_prompt_len, chunk_size):
            prompt_chunks = [p[s : s + chunk_size] for p in encoded_prompts]
            assert all(len(p) > 0 for p in prompt_chunks)

            input_ids = torch.tensor(
                sum(prompt_chunks, []), device=model.device, dtype=torch.long
            )
            chunk_seqlens = [len(p) for p in prompt_chunks]

            # Get activations for the target layer
            activations = model.get_acts(
                input_ids, chunk_seqlens, target_layer=target_layer, cache=None
            )
            layer_activations.append(activations)

        # Concatenate activations for the target layer
        all_activations = torch.cat(layer_activations, dim=0)

        return all_activations
