from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    model_name: str
    local_path: str
    huggingface_id: str
    vocab_size: int
    hidden_size: int

MODEL_CONFIGS = {
    "mistral-7b": ModelConfig(
        model_name="mistral-7b",
        local_path="Mistral-7B-Instruct-v0.3",
        huggingface_id="mistralai/Mistral-7B-Instruct-v0.3",
        vocab_size=131072,
        hidden_size=4096
    ),
    "pixtral-12b": ModelConfig(
        model_name="pixtral-12b",
        local_path="pixtral-12B-2409",
        huggingface_id="mistralai/Pixtral-12B-2409",
        vocab_size=131072
    )
}

def get_model_config(model_name: str) -> Optional[ModelConfig]:
    return MODEL_CONFIGS.get(model_name)
