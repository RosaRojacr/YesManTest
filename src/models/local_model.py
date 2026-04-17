import os
import sys
import gc
import torch
import yaml
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = "R:\\YesManTest"
_hf_authenticated = False


class LocalModel:
    def __init__(self, config=None, config_path=None):
        global _hf_authenticated
        
        # Environment setup
        os.chdir(PROJECT_ROOT)
        if PROJECT_ROOT not in sys.path:
            sys.path.insert(0, PROJECT_ROOT)
        
        # Load config
        if config is not None:
            self.config = config
        else:
            if config_path is None:
                config_path = os.path.join(PROJECT_ROOT, "config", "default.yaml")
            with open(config_path) as f:
                self.config = yaml.safe_load(f).get("model", {})
        
        # Point HF cache to R drive before anything downloads
        os.environ["HF_HOME"] = self.config.get("cache_dir", os.path.join(PROJECT_ROOT, "Model Cache"))
        
        # Authenticate once per session
        if not _hf_authenticated:
            token_path = os.path.join(PROJECT_ROOT, "Authentication", "HF_Token.txt")
            with open(token_path) as f:
                login(token=f.read().strip())
            _hf_authenticated = True
        
        # Load model
        print(f"Initializing load for: {self.config['name']}")
        
        cache_dir = self.config.get("cache_dir")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["name"],
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["name"],
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            ),
            device_map={"": 0},
            low_cpu_mem_usage=True,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        print(f"Model loaded successfully: {self.config['name']}")

    def ask(self, system_prompt, user_message, max_new_tokens=None):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        if self.tokenizer.chat_template:
            try:
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                # Some models (e.g. Gemma) don't support system role — fold it into user message
                messages = [{"role": "user", "content": f"{system_prompt}\n\n{user_message}"}]
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"{system_prompt}\n\nUser: {user_message}\nAssistant:"
        
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        input_length = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.config.get("max_new_tokens", 300),
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

    def cleanup(self):
        if hasattr(self, 'model'):
            self.model.to("cpu")
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print("VRAM cleared.")