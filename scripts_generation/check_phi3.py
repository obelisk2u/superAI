import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

def load_model():
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
        trust_remote_code=False,   # disable remote code
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_cfg,
        device_map="auto",
        trust_remote_code=False,   # disable remote code
    )

    # Optionally disable use_cache if you run into other caching errors
    model.config.use_cache = False

    return tokenizer, model

def quick_test(tokenizer, model):
    prompt = "user1: hey guys\nuser2: PogChamp\nuser3:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            temperature=0.9,
            top_p=0.95,
            do_sample=True,
        )

    print("\n=== OUTPUT ===\n")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    tok, mdl = load_model()
    quick_test(tok, mdl)