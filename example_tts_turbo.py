import torchaudio as ta
import torch
import intel_extension_for_pytorch as ipex
from chatterbox.tts_turbo import ChatterboxTurboTTS

# Load the Turbo model
model = ChatterboxTurboTTS.from_pretrained(device="xpu")

# Intel XPU Optimization
# 'ipex.optimize' fuses kernels (like Linear+ReLU) into single GPU calls
model.t3 = ipex.optimize(model.t3, dtype=torch.float16)
model.s3gen = ipex.optimize(model.s3gen, dtype=torch.float16)
model.ve = ipex.optimize(model.ve, dtype=torch.float16)

# Generate with Paralinguistic Tags
text = "Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. It's the SkyNet T-800 series and it's got basically everything. Including AI integration with ChatGPT and all that jazz. Would you like me to get some prices for you?"

with torch.autocast(device_type="xpu", enabled=True, dtype=torch.float16):
    # Generate audio (requires a reference clip for voice cloning)
    # wav = model.generate(text, audio_prompt_path="your_10s_ref_clip.wav")
    wav = model.generate(text)

# Save the output
ta.save("test-turbo.wav", wav, model.sr)
