import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
import webrtcvad
import struct
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model     = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.eval()


def apply_vad(audio: np.ndarray, sr: int, aggressiveness: int = 2) -> np.ndarray:  #aggressiveness: 0-3 arası, 0 ən az, 3 ən çox sükutu kəsəcək
    frame_ms = 30
    frame_size = int(sr * frame_ms / 1000)
    audio_int16 = (audio * 32768).astype(np.int16)
    vad = webrtcvad.Vad(aggressiveness)
    voiced_frames = []
    for i in range(0, len(audio_int16) - frame_size, frame_size):
        frame = audio_int16[i : i + frame_size]
        raw = struct.pack("%dh" % len(frame), *frame)
        if vad.is_speech(raw, sr):
            voiced_frames.append(audio[i : i + frame_size])
    if not voiced_frames:
        return audio
    return np.concatenate(voiced_frames)


def reduce_noise(audio: np.ndarray, sr: int) -> np.ndarray:
    reduced = nr.reduce_noise(y=audio, sr=sr, stationary=True) #audio-nun içindəki səs-küyü azaltır, stationary=True isə səs-küyün zamanla dəyişmədiyini göstərir
    return reduced


def normalize(audio: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(audio))
    if max_val == 0:
        return audio
    return audio / max_val



def preprocess(audio_path: str) -> np.ndarray:
    # 1. Yüklə
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    print(f"Orijinal uzunluq: {len(audio)/sr:.2f} san")

    # 2. VAD — əvvəlcə sükutu kəsirik
    audio = apply_vad(audio, sr, aggressiveness=2)
    print(f"VAD sonrası uzunluq: {len(audio)/sr:.2f} san")

    # 3. Noise reduction — küyü azaldiriq
    audio = reduce_noise(audio, sr)
    print("Noise reduction tamamlandı")

    # 4. Normalizasiya — səviyyəni standartlaşdıririq
    audio = normalize(audio)
    print("Normalizasiya tamamlandı")

    return audio

def transcribe(audio: np.ndarray):
    # 2. Chunk-lara böluruk
    chunk_size = 30 * 16000  # 30 saniyə × 16000 Hz = 480,000 nümunə

    chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]

    full_text = ""
    for chunk in chunks:
        inputs = processor(chunk, sampling_rate=16000, return_tensors="pt",return_attention_mask=True)
        with torch.no_grad():
            ids = model.generate(
                inputs["input_features"],
                attention_mask=inputs["attention_mask"],
                language="en",
                task="transcribe"
            )
        full_text += processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        
    return full_text


if __name__ == "__main__":
    audio_file = r"C:\Users\chichak.asgarova\Documents\voice_detection\audio\arctic_a0001_1592748574.flac"
    audio_clean = preprocess(audio_file)
    sf.write("cleaned_audio.wav", audio_clean, 16000)
    print("Təmizlənmiş audio 'cleaned_audio.wav' faylına yazıldı")
    english = transcribe(audio_clean)
    print("=" * 50)
    print(f"English     : {english}")
    print("=" * 50) 