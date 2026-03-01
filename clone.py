# %%
import torch
import torchaudio
import librosa
import soundfile as sf
# This is the main class they want you to use
from qwen_tts import Qwen3TTSModel

# 1. Load the model (This will download files from Hugging Face if needed)
model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
# Load the model weights first
model = Qwen3TTSModel.from_pretrained(model_id, device_map="cuda")


print("Model loaded successfully!")

# %%
# 1. Provide your 3-second reference clip
ref_audio = "./steve-sample.wave"
ref_text = "Hello, this is a test of my audio. I want to try and clone something but I'm not exactly sure how well it's going to do it. So I figured I'd give this a shot. I'm not sure but we'll see if it works well or it's really terrible."
# 2. Provide the new text you want to speak
target_text = "Now I am speaking locally with Qwen3-TTS!"

ref_audio, sr = torchaudio.load(ref_audio)
#ref_audio, sr = librosa.load(ref_audio, sr=None)


#%%
# 3. Generate the clone
print("Generating clone...")
wav, sr = model.generate_voice_clone(
    text=target_text,
    ref_audio=ref_audio,
    ref_text=ref_text
)

# 4. Save your voice output
sf.write("cloned_output.wav", wav, sr)
print("✅ Done! Saved to cloned_output.wav")


#%%
# Extract the embedding from your 3-second sample
speaker_embedding = model.extract_speaker_embedding(ref_audio, ref_text)

# Save it as a small file (like a .pt or .bin)
torch.save(speaker_embedding, "my_personal_voice.pt")
print("Voice profile saved!")

#%%
# Load your saved voice print
saved_voice = torch.load("my_personal_voice.pt")

# Generate using the embedding instead of the ref_audio
wav, sr = model.generate(
    text="Using my saved voice profile now!",
    speaker_embedding=saved_voice
)