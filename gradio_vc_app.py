import torch
import gradio as gr
from chatterbox.vc import ChatterboxVC
import intel_extension_for_pytorch as ipex

DEVICE = "xpu" if torch.xpu.is_available() else "cpu"


model = ChatterboxVC.from_pretrained(DEVICE)

# Intel XPU Optimization
# 'ipex.optimize' fuses kernels (like Linear+ReLU) into single GPU calls
model = ipex.optimize(model, dtype=torch.float16)


def generate(audio, target_voice_path):
    wav = model.generate(
        audio,
        target_voice_path=target_voice_path,
    )
    return model.sr, wav.squeeze(0).numpy()


demo = gr.Interface(
    generate,
    [
        gr.Audio(
            sources=["upload", "microphone"], type="filepath", label="Input audio file"
        ),
        gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",
            label="Target voice audio file (if none, the default voice is used)",
            value=None,
        ),
    ],
    "audio",
)

if __name__ == "__main__":
    demo.launch()
