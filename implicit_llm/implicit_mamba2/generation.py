
from typing import Any, Tuple
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import Tensor



class GenerationMixin:
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        raise NotImplementedError

    @torch.inference_mode()
    def generate(
        self,
        prompt: Tensor,
        max_len: int,
        temperature: float = 1.0,
        inference_cache: Any = None,
        return_only_generated: bool = False,
        verbose: bool = False,
    ) -> Tuple[Tensor, Any]:
        """
        Sequentially generate text from a prompt.
        Args:
            prompt: Input tokens to start the generation from (B, L) in integer format
            max_len: Maximum generation length
            temperature: Parameter to control the randomness of the generation
            inference_cache: Cache of previous hidden states
            return_only_generated: Return only the generated tokens, not the prompt

        Returns:
            generated: Generated tokens (B, L) in integer format
            inference_cache: Updated inference cache
        """
        if prompt.size(0) != 1:
            raise ValueError(
                f"Batch generation is not supported. Expected batch size 1, got {prompt.size(0)}. "
                "Generate one sequence at a time."
            )
        # make sure we are in evaluation mode
        self.eval()

        def decode_token(y):
            logits = self.criterion.decoder(y)
            if temperature == 0.0:
                return logits.argmax(dim=-1, keepdim=True)
            probs = F.softmax(logits / temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1)

        def iterator(length, tag=""):
            return tqdm(range(length), tag) if verbose else range(length)

        if inference_cache is None:
            inference_cache = self.backbone.allocate_inference_cache(
                prompt.size(0), self.word_emb.weight.device, dtype=torch.float32
            )

        # prefill
        xs = self.word_emb(prompt)
        for t in iterator(prompt.size(1), "prefill"):
            out, inference_cache, _ = self.backbone.generation_step(xs[:, t], inference_cache)
        # start generation with last predicted token from prompt
        input_token = decode_token(out)
        tokens = [input_token]
        # generate remaining tokens
        for _ in iterator(max_len - 1, "generation"):
            # forward pass
            x = self.word_emb(input_token).squeeze(1)
            out, inference_cache, _ = self.backbone.generation_step(x, inference_cache)
            input_token = decode_token(out)
            tokens.append(input_token)
        generated = torch.cat(tokens, dim=-1)

        if return_only_generated:
            return generated, inference_cache
        else:
            return torch.cat([prompt, generated], dim=-1), inference_cache
