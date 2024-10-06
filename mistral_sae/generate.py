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
    top_p: float = 0.9,
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
            using_sae=True,
            sae=sae,
            features=features,
        )

        logits = torch.log_softmax(prelogits, dim=-1)

        # Integrate SAE model outputs if generating
        if sae is not None and is_generating:
            # Obtain SAE activations
            with torch.no_grad():
                sae_activations = sae(input_tensor.unsqueeze(0))
                # Assume sae_activations is of shape [1, num_features, seq_len]
                # You may need to process sae_activations to match logits shape
                # Example: Map SAE activations to vocab logits
                # This is a placeholder and should be replaced with your actual logic
                sae_adjustment = sae_activations.squeeze(0).mean(dim=0)
                sae_adjustment = sae_adjustment[-1]  # Take activation for the last token
                sae_adjustment = sae_adjustment.unsqueeze(0).expand_as(logits)
                # Adjust logits with SAE activations
                logits += sae_adjustment  # Adjust logits (modify as needed)

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
