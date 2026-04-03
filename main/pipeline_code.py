import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from transformers import pipeline

# Pipeline qur
asr = pipeline(
    "automatic-speech-recognition",# this is the task name for speech-to-text
    model="openai/whisper-small",
    device=0 if torch.cuda.is_available() else -1  # if you have GPU(0),run this model on it ,it you dont have run it on CPU(-1)
)

# Funksiya
def transcribe(audio_path: str):
        # librosa ilə oxu — ffmpeg lazım deyil
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)

    result = asr(
        {"array": audio, "sampling_rate": 16000},
        generate_kwargs={"language": "en", "task": "transcribe"}
    )
    english_text = result["text"].strip()
    return english_text


# İşlət
if __name__ == "__main__":
    audio_file = r"C:\Users\chichak.asgarova\Documents\voice_detection\audio\arctic_a0001_1592748574.flac"
    english = transcribe(audio_file)
    print("=" * 50)
    print(f"English     : {english}")
    print("=" * 50) 