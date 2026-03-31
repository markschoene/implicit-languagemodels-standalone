import torch

from . import ImplicitLlamaForCausalLM, ImplicitMambaForCausalLM
from .implicit_llama.modeling_llama import (
    ImplicitCausalLMOutputWithPast as ImplicitCausalLMOutputWithPastLlama,
)
from .implicit_mamba2.modeling_mamba2 import (
    ImplicitCausalLMOutputWithPast as ImplicitCausalLMOutputWithPastMamba,
)


def sequential_forward(
    model: ImplicitLlamaForCausalLM | ImplicitMambaForCausalLM,
    input_ids: torch.LongTensor,
) -> ImplicitCausalLMOutputWithPastLlama | ImplicitCausalLMOutputWithPastMamba:
    """
    Helper function to sequentially step though a sequence and compute logits for each token.
    """
    model.eval()
    with torch.no_grad():
        if isinstance(model, ImplicitMambaForCausalLM):
            output = sequential_forward_mamba(model, input_ids=input_ids)
        elif isinstance(model, ImplicitLlamaForCausalLM):
            output = sequential_forward_llama(model, input_ids=input_ids)
        else:
            raise ValueError("Model type not supported for sequential evaluation.")
    return output


def sequential_forward_mamba(
    model: ImplicitMambaForCausalLM, input_ids: torch.LongTensor,
) -> ImplicitCausalLMOutputWithPastMamba:
    """
    Perform a sequential forward pass with a Mamba state-space model.
    """
    logits = []
    last_hidden_states = []
    implicit_metrics = []

    def append(x):
        logits.append(x.logits)
        last_hidden_states.append(x.last_hidden_state.squeeze(1))
        implicit_metrics.append(x.implicit_metrics)

    # first step to initialize the cache
    out = model(input_ids=input_ids[:, 0].unsqueeze(1), use_cache=True)
    append(out)

    cache_params = out.cache_params

    # iterate the sequence
    for t in range(1, input_ids.size(1)):
        out = model(
            input_ids=input_ids[:, t],
            use_cache=True,
            cache_params=cache_params,
            cache_position=torch.LongTensor([t]).to(input_ids.device),
        )
        append(out)
    
    # Concatenate logits and other outputs
    logits = torch.cat(logits, dim=1)
    last_hidden_states = torch.stack(last_hidden_states, dim=1)
    
    # average implicit metrics across the sequence
    implicit_metrics = {
        key: torch.stack([metric[key] for metric in implicit_metrics]).mean()
        for key in implicit_metrics[0].keys()
    }
    
    output = ImplicitCausalLMOutputWithPastMamba(
        loss=None,  # Loss is not computed in this forward pass
        logits=logits,
        cache_params=cache_params,
        last_hidden_state=last_hidden_states,
        implicit_metrics=implicit_metrics,
        jac_loss=None,
    )
    return output


def sequential_forward_llama(
    model: ImplicitLlamaForCausalLM, input_ids: torch.LongTensor,
) -> ImplicitCausalLMOutputWithPastLlama:
    """
    Perform a sequential forward pass with a Llama transformer model.
    """
    logits = []
    hidden_states = []
    implicit_metrics = []
    attentions = []

    def append(x):
        logits.append(x.logits)
        if x.hidden_states is not None:
            hidden_states.append(x.hidden_states)
        if x.attentions is not None:
            attentions.append(x.attentions)
        implicit_metrics.append(x.implicit_metrics)

    # first step to initialize the cache
    out = model(input_ids=input_ids[:, 0].unsqueeze(1), use_cache=True)
    append(out)

    # the cache is updated in place
    past_key_values = out.past_key_values

    # iterate the sequence
    for t in range(1, input_ids.size(1)):
        out = model(
            input_ids=input_ids[:, t].unsqueeze(1),
            use_cache=True,
            past_key_values=past_key_values,
            cache_position=torch.LongTensor([t]).to(input_ids.device),
        )
        append(out)

    # Concatenate logits and other outputs
    logits = torch.cat(logits, dim=1)
    if hidden_states:
        hidden_states = torch.stack(hidden_states, dim=1)
    else:
        hidden_states = None
    if attentions:
        attentions = torch.stack(attentions, dim=1)
    else:
        attentions = None
    
    # average implicit metrics across the sequence
    implicit_metrics = {
        key: torch.stack([metric[key] for metric in implicit_metrics]).mean()
        for key in implicit_metrics[0].keys()
    }    

    output = ImplicitCausalLMOutputWithPastLlama(
        loss=None,  # Loss is not computed in this forward pass
        logits=logits,
        past_key_values=past_key_values,
        hidden_states=hidden_states,
        attentions=attentions,
        implicit_metrics=implicit_metrics,
        jac_loss=None,
    )
    return output