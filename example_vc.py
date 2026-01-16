import torch
import torchaudio as ta

from chatterbox.vc import ChatterboxVC
import intel_extension_for_pytorch as ipex

# Automatically detect the best available device
if torch.xpu.is_available():
    device = "xpu"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

AUDIO_PATH = "YOUR_FILE.wav"
TARGET_VOICE_PATH = "YOUR_FILE.wav"

model = ChatterboxVC.from_pretrained(device)

# Intel XPU Optimization
# 'ipex.optimize' fuses kernels (like Linear+ReLU) into single GPU calls
model = ipex.optimize(model, dtype=torch.float16)

wav = model.generate(
    audio=AUDIO_PATH,
    target_voice_path=TARGET_VOICE_PATH,
)
ta.save("testvc.wav", wav, model.sr)
