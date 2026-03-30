# register 'implicit_causal_lm' local models to automodel
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.utils import is_liger_kernel_available

from .implicit_llama.configuration_llama import ImplicitLlamaConfig
from .implicit_llama.modeling_llama import ImplicitLlamaForCausalLM
from .implicit_mamba2.configuration_mamba2 import ImplicitMambaConfig
from .implicit_mamba2.modeling_mamba2 import ImplicitMambaForCausalLM

if is_liger_kernel_available():
    from liger_kernel.transformers import AutoLigerKernelForCausalLM


def register_implicit_causal_lm() -> None:
    """
    Register the 'implicit_causal_lm' model locally to AutoModel, AutoModelForCausalLM, and AutoConfig.
    This allows the use of AutoModel.from_pretrained() with the implicit models.
    """
    AutoConfig.register("implicit_llama3", ImplicitLlamaConfig)
    AutoModel.register(ImplicitLlamaConfig, ImplicitLlamaForCausalLM)
    AutoModelForCausalLM.register(ImplicitLlamaConfig, ImplicitLlamaForCausalLM)
    if is_liger_kernel_available():
        AutoLigerKernelForCausalLM.register(ImplicitLlamaConfig, ImplicitLlamaForCausalLM)
    # register the implicit mamba model
    AutoConfig.register("implicit_mamba2", ImplicitMambaConfig)
    AutoModel.register(ImplicitMambaConfig, ImplicitMambaForCausalLM)
    AutoModelForCausalLM.register(ImplicitMambaConfig, ImplicitMambaForCausalLM)
    if is_liger_kernel_available():
        AutoLigerKernelForCausalLM.register(ImplicitMambaConfig, ImplicitMambaForCausalLM)
