import torch
from transformers import AutoModel, AutoTokenizer

import implicit_llm

prompt = "What is the capital of France? The"


def test_generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate text using the model
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=inputs.input_ids.shape[1] + 10,
            do_sample=False,
            pad_token_id=model.config.eos_token_id,
        )

    # Decode the generated tokens
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def main(model_type, size, backbone):
    model_name = f"hf_models/{backbone}-{size}-{model_type}"
    model = AutoModel.from_pretrained(model_name)
    model = model.to("cuda")

    # prepare evaluation
    model.eval()
    if model.config.backbone_type != "explicit":
        model.backbone.sequential_evaluation()

    tokenizer = AutoTokenizer.from_pretrained(model.tokenizer)

    text = test_generate(model, tokenizer, prompt)
    print(40 * "=")
    print(f"Model: {model_name}")
    print(40 * "=")
    print(text)


if __name__ == "__main__":
    for size in ["130m", "370m", "760m", "1.3b"]:
        for model_type in ["explicit", "pretrain", "implicit"]:
            for backbone in ["mamba2", "llama3"]:
                main(model_type, size, backbone)
