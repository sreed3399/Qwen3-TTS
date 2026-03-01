#%%

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel


model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    #dtype=torch.bfloat16,
    #attn_implementation="flash_attention_2",
)

print("Model loaded successfully!")

# %%
ref_audio = "./steve-sample.wav"
ref_text  = "Hello, this is a test of my audio. I want to try and clone something but I'm not exactly sure how well it's going to do it. So I figured I'd give this a shot. I'm not sure but we'll see if it works well or it's really terrible."

print("Generating clone...")

wavs, sr = model.generate_voice_clone(
    text="Hello, Mahal! I am speaking with a cloned voice now. I hope you enjoy my lovely, velvety tones. Love you!",
    language="English",
    ref_audio=ref_audio,
    ref_text=ref_text,
)

sf.write("output_voice_clone.wav", wavs[0], sr)
print("✅ Done! Saved to output_voice_clone.wav")