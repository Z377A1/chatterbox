import torchaudio as ta
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS

# Load the Turbo model
model = ChatterboxTurboTTS.from_pretrained(device="xpu")

# Generate with Paralinguistic Tags
text = "Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. It's the SkyNet T-800 series and it's got basically everything. Including AI integration with ChatGPT and all that jazz. Would you like me to get some prices for you?"

with torch.autocast(device_type="xpu", enabled=True, dtype=torch.float16):
    # Generate audio (requires a reference clip for voice cloning)
    # wav = model.generate(text, audio_prompt_path="your_10s_ref_clip.wav")
    wav = model.generate(text)

# Save the output
ta.save("test-turbo.wav", wav, model.sr)
