import os
os.add_dll_directory(r"C:\Users\jv\MOSS-TTS")

import torch
import torchaudio
from transformers import AutoModel, AutoProcessor, GenerationConfig
import gradio as gr
from datetime import datetime
import importlib.util
import traceback
import gc
import time
import atexit
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Memory optimization settings for GPU
torch.backends.cuda.enable_cudnn_sdp(True)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

class DelayGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layers = kwargs.get("layers", [{} for _ in range(32)])
        self.do_samples = kwargs.get("do_samples", None)
        self.n_vq_for_inference = 32

# Global variables
model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# CAMINHO CORRIGIDO: Usando barras normais (/) para evitar o bug de Repo ID no Windows
MODEL_PATH = "C:/Users/jv/MOSS-TTS/moss_model"

def cleanup_model():
    """Unload model from GPU memory"""
    global model, processor
    if model is not None:
        print("🧹 Cleaning up model from GPU...")
        del model
        model = None
    if processor is not None:
        if hasattr(processor, 'audio_tokenizer'):
            del processor.audio_tokenizer
        del processor
        processor = None
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
        print("✅ GPU memory cleared!")

atexit.register(cleanup_model)

def resolve_attn_implementation() -> str:
    if device == "cuda":
        return "sdpa"
    return "eager"

def load_model():
    """Load model with optimized settings using local paths"""
    global model, processor

    if model is None:
        print("🔄 Loading MOSS-TTS from local folder...")

        attn_implementation = resolve_attn_implementation()
        print(f"Using attention: {attn_implementation}")

        # Carregando o Processor localmente
        processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )

        processor.audio_tokenizer = processor.audio_tokenizer.to(device)

        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        # Carregando o Modelo localmente
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)

        model.eval()

        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            vram = torch.cuda.memory_allocated() / 1024**3
            print(f"✅ Model loaded! VRAM: {vram:.2f}GB")

    return model, processor

PRESETS = {
    "Fast (8 RVQ)": {
        "n_vq": 8,
        "text_temp": 1.5,
        "audio_temp": 0.95,
        "text_top_p": 1.0,
        "audio_top_p": 0.95,
        "text_top_k": 50,
        "audio_top_k": 50,
        "audio_rep_pen": 1.1
    },
    "Balanced (16 RVQ)": {
        "n_vq": 16,
        "text_temp": 1.5,
        "audio_temp": 0.95,
        "text_top_p": 1.0,
        "audio_top_p": 0.95,
        "text_top_k": 50,
        "audio_top_k": 50,
        "audio_rep_pen": 1.1
    },
    "High Quality (24 RVQ)": {
        "n_vq": 24,
        "text_temp": 1.5,
        "audio_temp": 0.95,
        "text_top_p": 1.0,
        "audio_top_p": 0.95,
        "text_top_k": 50,
        "audio_top_k": 50,
        "audio_rep_pen": 1.1
    },
    "Maximum (32 RVQ)": {
        "n_vq": 32,
        "text_temp": 1.5,
        "audio_temp": 0.95,
        "text_top_p": 1.0,
        "audio_top_p": 0.95,
        "text_top_k": 50,
        "audio_top_k": 50,
        "audio_rep_pen": 1.1
    }
}

def apply_preset(preset_name):
    """Return preset values"""
    preset = PRESETS[preset_name]
    return (
        preset["n_vq"],
        preset["text_temp"],
        preset["text_top_p"],
        preset["text_top_k"],
        preset["audio_temp"],
        preset["audio_top_p"],
        preset["audio_top_k"],
        preset["audio_rep_pen"]
    )

def generate_speech(
    text,
    reference_audio,
    max_new_tokens,
    speed,
    text_temp,
    text_top_p,
    text_top_k,
    audio_temp,
    audio_top_p,
    audio_top_k,
    audio_repetition_penalty,
    n_vq,
    progress=gr.Progress()
):
    """Generate TTS with memory-efficient long-form generation"""

    if not text or len(text.strip()) == 0:
        return None, "⚠️ Please enter text!"

    try:
        os.makedirs("outputs", exist_ok=True)

        progress(0, desc="Loading model...")
        model, processor = load_model()

        text_length = len(text)
        estimated_duration = max_new_tokens / 12.5

        status = f"📝 Text: {text_length:,} chars\n"
        status += f"🎯 Target: {max_new_tokens} tokens (~{estimated_duration/60:.1f} min)\n\n"

        yield None, status

        # Build conversation
        progress(0.1, desc="Processing...")
        if reference_audio is not None:
            ref_audio_path = os.path.abspath(reference_audio).replace("\\", "/")
            status += f"🎙️ Voice cloning: {os.path.basename(ref_audio_path)}\n"
            conversations = [[
                processor.build_user_message(text=text, reference=[ref_audio_path])
            ]]
        else:
            status += "🎙️ Default voice\n"
            conversations = [[
                processor.build_user_message(text=text)
            ]]

        yield None, status

        # Process input
        batch = processor(conversations, mode="generation")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Fix temperature bug
        if text_temp == 1.0:
            text_temp = 1.001
        if audio_temp == 1.0:
            audio_temp = 1.001

        # Generation config
        generation_config = DelayGenerationConfig()
        generation_config.pad_token_id = processor.tokenizer.pad_token_id
        generation_config.eos_token_id = 151653
        generation_config.max_new_tokens = max_new_tokens
        generation_config.use_cache = True
        generation_config.do_sample = True
        generation_config.num_beams = 1

        generation_config.n_vq_for_inference = n_vq
        generation_config.do_samples = [True] * (n_vq + 1)
        generation_config.layers = [
            {
                "repetition_penalty": 1.0,
                "temperature": text_temp,
                "top_p": text_top_p,
                "top_k": text_top_k
            }
        ] + [
            {
                "repetition_penalty": audio_repetition_penalty,
                "temperature": audio_temp,
                "top_p": audio_top_p,
                "top_k": audio_top_k
            }
        ] * n_vq

        # Clear cache
        progress(0.2, desc="Clearing cache...")
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        status += f"\n🎵 Generating...\n"
        yield None, status

        # Generate
        start_time = time.time()
        progress(0.3, desc="Generating...")

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config
            )

        gen_time = time.time() - start_time

        progress(0.85, desc="Decoding...")
        status += f"✅ Generated in {gen_time:.1f}s\n"
        status += "🔊 Decoding...\n"
        yield None, status

        # Decode
        decoded_messages = processor.decode(outputs)
        audio = decoded_messages[0].audio_codes_list[0]

        # Clear memory
        if device == "cuda":
            del outputs, input_ids, attention_mask, batch, decoded_messages
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()

        # Speed
        progress(0.94, desc="Speed adjust...")
        if speed != 1.0:
            sample_rate = processor.model_config.sampling_rate
            new_sample_rate = int(sample_rate * speed)
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=new_sample_rate
            )
            audio_resampled = resampler(audio.unsqueeze(0)).squeeze(0)
            resampler_back = torchaudio.transforms.Resample(
                orig_freq=new_sample_rate,
                new_freq=sample_rate
            )
            audio = resampler_back(audio_resampled.unsqueeze(0)).squeeze(0)

        progress(0.97, desc="Saving...")

        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"outputs/moss_tts_{timestamp}.wav"
        torchaudio.save(
            output_path,
            audio.unsqueeze(0),
            processor.model_config.sampling_rate
        )

        duration = len(audio) / processor.model_config.sampling_rate
        vram = torch.cuda.memory_allocated() / 1024**3 if device == "cuda" else 0
        rtf = gen_time / duration if duration > 0 else 0

        progress(1.0, desc="Done!")

        status += f"\n🎉 SUCCESS!\n"
        status += f"📏 Audio: {duration:.1f}s ({duration/60:.2f} min)\n"
        status += f"⏱️ Generation: {gen_time:.1f}s ({gen_time/60:.1f} min)\n"
        status += f"🚀 RTF: {rtf:.2f}x\n"
        status += f"🎚️ Speed: {speed}x\n"
        status += f"📊 VRAM: {vram:.2f}GB\n"
        status += f"🎛️ RVQ: {n_vq}/32\n"
        status += f"💾 {output_path}"

        yield output_path, status

    except torch.cuda.OutOfMemoryError as e:
        error_msg = f"❌ OUT OF MEMORY!\n\n"
        error_msg += f"Tried: {max_new_tokens} tokens with {n_vq} RVQ\n\n"
        error_msg += f"Solutions:\n"
        error_msg += f"1. Reduce Max Tokens\n"
        error_msg += f"2. Use Fast (8 RVQ) preset\n"
        error_msg += f"3. Click 'Clear GPU' and retry\n\n"
        yield None, error_msg
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        yield None, error_msg

# =============================================
# GRADIO INTERFACE
# =============================================

custom_css = """
.aiquest-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 16px;
    text-align: center;
    color: white;
}
.aiquest-header h1 {
    margin: 0 0 8px 0;
    font-size: 1.8em;
    color: white !important;
}
.aiquest-header p {
    margin: 4px 0;
    opacity: 0.95;
    color: white !important;
}
"""

with gr.Blocks(title="MOSS-TTS Local", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.HTML("""
    <div class="aiquest-header">
        <h1>🎙️ MOSS-TTS 1.7B Zero-Shot Voice Cloning (LOCAL)</h1>
        <p>Rodando offline na sua máquina</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="📝 Text",
                placeholder="Paste your script here...",
                lines=10,
                value="Hello! This is MOSS text-to-speech, running locally on my PC."
            )
            reference_audio = gr.Audio(
                label="🎤 Reference Voice (Optional)",
                type="filepath",
                sources=["upload"]
            )
            preset_dropdown = gr.Dropdown(
                choices=list(PRESETS.keys()),
                value="Balanced (16 RVQ)",
                label="Preset"
            )
            with gr.Row():
                max_tokens = gr.Slider(50, 5000, 2500, step=100, label="Max Tokens")
                speed = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="Speed")

            with gr.Accordion("⚙️ Advanced Settings", open=False):
                n_vq = gr.Slider(8, 32, 8, step=1, label="RVQ Layers")
                with gr.Row():
                    text_temp = gr.Slider(0.1, 2.0, 1.5, step=0.1, label="Text Temp")
                    text_top_p = gr.Slider(0.1, 1.0, 1.0, step=0.05, label="Text Top-P")
                    text_top_k = gr.Slider(1, 100, 50, step=1, label="Text Top-K")
                with gr.Row():
                    audio_temp = gr.Slider(0.1, 2.0, 0.95, step=0.05, label="Audio Temp")
                    audio_top_p = gr.Slider(0.1, 1.0, 0.95, step=0.05, label="Audio Top-P")
                with gr.Row():
                    audio_top_k = gr.Slider(1, 100, 50, step=1, label="Audio Top-K")
                    audio_rep_pen = gr.Slider(1.0, 1.5, 1.1, step=0.05, label="Rep Penalty")

            with gr.Row():
                generate_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg", scale=3)
                clear_btn = gr.Button("🧹 Clear GPU", variant="secondary", size="lg", scale=1)

        with gr.Column(scale=1):
            audio_output = gr.Audio(label="🔊 Generated Audio", type="filepath")
            status_output = gr.Textbox(label="📊 Status", lines=16, interactive=False)

    preset_dropdown.change(
        fn=apply_preset,
        inputs=[preset_dropdown],
        outputs=[n_vq, text_temp, text_top_p, text_top_k,
                audio_temp, audio_top_p, audio_top_k, audio_rep_pen]
    )

    generate_btn.click(
        fn=generate_speech,
        inputs=[text_input, reference_audio, max_tokens, speed,
                text_temp, text_top_p, text_top_k,
                audio_temp, audio_top_p, audio_top_k,
                audio_rep_pen, n_vq],
        outputs=[audio_output, status_output]
    )

    def clear_memory():
        cleanup_model()
        return "✅ GPU cleared! Ready for next generation."

    clear_btn.click(fn=clear_memory, inputs=[], outputs=[status_output])

print("✅ MOSS-TTS ready! Launching local Gradio...")
demo.launch(inbrowser=True)