import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image
import torch
import matplotlib.pyplot as plt
from transformers import logging as tf_logging

# Ensure bbq root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

tf_logging.set_verbosity_error()

from bbq import (
    ModelConfigWrapper,
    load_configuration_from_yaml_file,
    EngineWrapper,
    initialize_engine_from_yaml_config,
)


def inspect_what_model_sees(
    processor: Any,
    batch: Dict[str, Any],
    original_image: Image.Image,
    embedding_result: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
) -> None:
    input_ids = batch["input_ids"][0].cpu().numpy()
    pv_raw = batch["pixel_values"][0]
    num_crops = pv_raw.shape[0] if pv_raw.dim() == 4 else 1

    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(getattr(processor, "config", None), "image_token_id", None)

    img_count = sum(1 for tid in input_ids if tid == image_token_id) if image_token_id else 0

    print(f"\n{'='*70}")
    print(f"INPUT INSPECTION | image_token_id={image_token_id} | seq_len={len(input_ids)} | sub_crops={num_crops}")
    print(f"{'='*70}")
    print(f"Image tokens: {img_count} | Text tokens: {len(input_ids) - img_count} | Sub-crop tiles: {num_crops}")

    if embedding_result is not None:
        print(f"\n{'='*70}")
        print("RETRIEVAL OUTPUT (Embedding vector for MaxSim scoring)")
        print(f"{'='*70}")
        print(f"Shape : {list(embedding_result.shape)}")
        print(f"Mean  : {embedding_result.mean().item():.4f}")
        print(f"Norm  : {embedding_result.norm().item():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(original_image)
    axes[0].set_title("Original Document Image", fontweight="bold")
    axes[0].axis("off")

    global_crop = pv_raw[-1] if pv_raw.dim() == 4 else pv_raw
    global_crop = global_crop.cpu().float()
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    global_crop = torch.clamp(global_crop * std + mean, 0, 1)
    axes[1].imshow(global_crop.permute(1, 2, 0).numpy())
    axes[1].set_title(f"Processed Model View ({num_crops} tiles)", fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved inspection report to: {save_path}")
    plt.close(fig)


def test_yaml_config_loading(config: ModelConfigWrapper) -> None:
    print(f"\n[Test] Loaded YAML Configuration:")
    print(f" base_model_id : {config.base_model_id}")
    print(f" lora_adapter_id : {config.lora_adapter_id}")
    print(f" embedding_dim : {config.embedding_dim}")
    print(f" device : {config.device}")
    assert isinstance(config.base_model_id, str) and len(config.base_model_id) > 0
    print("[Test] test_yaml_config_loading PASSED!")


def test_process_texts_shapes(engine: EngineWrapper) -> None:
    texts = ["revenue chart", "invoice"]
    batch = engine.processor.process_texts(texts)
    print(f"\n[Test] process_texts inputs: {texts}")
    print(f"[Test] input_ids shape: {batch['input_ids'].shape}")
    assert batch["input_ids"].shape[0] == 2
    assert "pixel_values" not in batch
    print("[Test] test_process_texts_shapes PASSED!")


def test_model_embeddings_output(engine: EngineWrapper) -> None:
    texts = ["revenue chart", "financial report"]
    embeddings = engine.encode_query_text_inputs(texts)

    print("\n" + "=" * 65)
    print(" ENGINE MODEL OUTPUT EMBEDDINGS")
    print("=" * 65)
    print(f"Model ({engine.config.base_model_id}):")
    print(f" - Output Shape : {list(embeddings.shape)}")
    print(f" - Vector Norm : {embeddings.norm(dim=-1).mean().item():.4f}")
    print("=" * 65)
    print("[Test] test_model_embeddings_output PASSED!")


def test_image_inspection(engine: EngineWrapper) -> None:
    image_path = "notebooks/sample.jpg"

    if not Path(image_path).exists():
        print(f"\n[Test] Sample image not found at {image_path}, skipping image inspection test.")
        return

    img = Image.open(image_path).convert("RGB")
    embeddings = engine.encode_multimodal_document_images([img])
    batch = engine.processor.process_images([img], prompt_command=engine.config.visual_prompt_command)

    cache_dir = Path.home() / ".cache" / "colpali_rag" / "data"
    save_path = str(cache_dir / "what_model_sees.png")

    inspect_what_model_sees(
        engine.processor,
        batch,
        img,
        embedding_result=embeddings[0],
        save_path=save_path,
    )
    print("[Test] test_image_inspection PASSED!")


def main() -> None:
    config_path = "config.yaml"
    print("=" * 65)
    print(" Initializing bbq engine (Loading model weights ONCE)...")
    print("=" * 65)
    config = load_configuration_from_yaml_file(config_path)
    engine = initialize_engine_from_yaml_config(config_path)
    print(" Engine loaded successfully!")
    print("=" * 65)

    test_yaml_config_loading(config)
    test_process_texts_shapes(engine)
    test_model_embeddings_output(engine)
    test_image_inspection(engine)

    print("=" * 65)
    print(" All bbq engine tests completed successfully with single-pass loading!")
    print("=" * 65)


if __name__ == "__main__":
    main()
