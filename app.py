#!/usr/bin/env python3
"""
================================================================================
Z-GO UNIVERSAL ORCHESTRATOR 
================================================================================

Text-to-image generator using Z-Image-Turbo Q4_K_M (4-bit quantized).
Features adjustable steps and size for speed control.

================================================================================
"""

import os
import sys
import platform
import gc
from pathlib import Path
from typing import Optional
from PIL import Image

# Memory optimization for MPS
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = Path(__file__).parent.absolute()
WEIGHTS_DIR = SCRIPT_DIR / "weights"

# 4-bit quantized model (proven working)
MODEL_REPO = "jayn7/Z-Image-Turbo-GGUF"
MODEL_FILE = "z_image_turbo-Q4_K_M.gguf"
BASE_MODEL = "Tongyi-MAI/Z-Image-Turbo"

# Global engine (lazy loaded)
_engine = None
_engine_info = None

# ============================================================================
# Hardware Detection
# ============================================================================

def get_hardware_info() -> str:
    system = platform.system()
    machine = platform.machine()
    if system == "Darwin" and machine == "arm64":
        return "Apple Silicon (M-series)"
    return f"{system} ({machine})"

# ============================================================================
# Lazy Engine Loader
# ============================================================================

def get_engine():
    """Lazy load the engine on first use."""
    global _engine, _engine_info
    
    if _engine is not None:
        return _engine, _engine_info
    
    import torch
    from huggingface_hub import hf_hub_download
    from diffusers import ZImagePipeline, ZImageTransformer2DModel, GGUFQuantizationConfig
    
    print("\n" + "=" * 50)
    print("📦 Loading Model (First Time)")
    print("=" * 50)
    
    # Ensure weights exist
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    weight_path = WEIGHTS_DIR / MODEL_FILE
    
    if not weight_path.exists():
        print(f"   Downloading {MODEL_FILE} (~5GB)...")
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir=WEIGHTS_DIR,
        )
    else:
        print(f"   ✅ Model found: {MODEL_FILE}")
    
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
        _engine_info = "CUDA GPU"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        _engine_info = "Apple MPS"
    else:
        device = "cpu"
        _engine_info = "CPU"
    
    print(f"   Device: {_engine_info}")
    print("   Loading transformer...")
    
    # Load transformer with bfloat16 (stable)
    transformer = ZImageTransformer2DModel.from_single_file(
        str(weight_path),
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
    )
    
    print("   Loading pipeline...")
    
    pipeline = ZImagePipeline.from_pretrained(
        BASE_MODEL,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    
    if device != "cpu":
        pipeline = pipeline.to(device)
    
    # Enable memory optimizations
    pipeline.enable_attention_slicing("max")
    
    _engine = (pipeline, device)
    print("\n✅ Ready!")
    print("=" * 50 + "\n")
    
    return _engine, _engine_info

# ============================================================================
# Generation Function
# ============================================================================

def generate_image(prompt: str, steps: int, size: str) -> Optional[Image.Image]:
    """Generate an image with configurable steps and size."""
    import torch
    
    if not prompt or not prompt.strip():
        return None
    
    # Parse size
    try:
        width, height = map(int, size.split("x"))
    except:
        width, height = 768, 768
    
    print(f"\n🎨 Generating...")
    print(f"   Prompt: {prompt[:50]}...")
    print(f"   Size: {width}x{height}, Steps: {steps}")
    
    try:
        (pipeline, device), _ = get_engine()
        
        gen_device = device if device != "mps" else "cpu"
        
        result = pipeline(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=0.0,
            height=height,
            width=width,
            generator=torch.Generator(gen_device).manual_seed(42),
        )
        
        image = result.images[0]
        
        # Clear MPS cache and garbage collect
        if device == "mps":
            torch.mps.empty_cache()
        gc.collect()
        
        print("✅ Done!")
        return image
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        # Clear memory on error too
        gc.collect()
        if 'torch' in dir() and hasattr(torch, 'mps'):
            torch.mps.empty_cache()
        return None

# ============================================================================
# Gradio UI
# ============================================================================

def create_ui():
    import gradio as gr
    
    hardware = get_hardware_info()
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="Z-Go Orchestrator",
        css="""
            .banner {
                background: linear-gradient(90deg, #1a1a2e, #16213e);
                color: #00d4ff;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                font-family: monospace;
                margin-bottom: 20px;
            }
            .gen-btn { height: 55px !important; font-size: 1.1em !important; }
        """
    ) as demo:
        
        gr.Markdown(
            """
            # 🧠 Z-Go Universal Orchestrator
            ### Z-Image-Turbo (Q4 Quantized)
            ---
            """
        )
        
        gr.HTML(f"""
            <div class="banner">
                ⚡ {hardware} &nbsp;|&nbsp; 
                📦 Q4_K_M (5GB) &nbsp;|&nbsp;
                � Lazy Load
            </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="✍️ Prompt",
                    placeholder="A beautiful landscape...",
                    lines=3,
                )
                
                with gr.Row():
                    steps = gr.Slider(
                        minimum=4, maximum=12, value=8, step=1,
                        label="⚡ Steps (fewer = faster)",
                    )
                    size = gr.Dropdown(
                        choices=["512x512", "768x768", "1024x1024"],
                        value="768x768",
                        label="📐 Size",
                    )
                
                generate_btn = gr.Button("🎨 Generate", variant="primary", elem_classes="gen-btn")
                
                gr.Markdown(
                    """
                    **Speed Guide:**
                    - 512x512 + 4 steps = Fastest
                    - 768x768 + 6 steps = Balanced
                    - 1024x1024 + 8+ steps = Best quality
                    """
                )
            
            with gr.Column(scale=1):
                output = gr.Image(label="🖼️ Result", type="pil", show_download_button=True, height=450)
        
        generate_btn.click(fn=generate_image, inputs=[prompt, steps, size], outputs=[output])
        prompt.submit(fn=generate_image, inputs=[prompt, steps, size], outputs=[output])
    
    return demo

# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "=" * 50)
    print("  🧠 Z-GO ORCHESTRATOR")
    print("  Q4 Quantized • Lazy Load")
    print("=" * 50)
    print(f"⚡ Hardware: {get_hardware_info()}")
    print("📦 Model loads on first generate")
    print("-" * 50)
    
    demo = create_ui()
    demo.launch(inbrowser=True, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
