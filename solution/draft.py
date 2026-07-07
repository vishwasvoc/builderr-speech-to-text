import numpy as np
from faster_whisper import WhisperModel
import threading

# Initialize the model once and keep it warm in memory
MODEL_ID = "small"
model = WhisperModel(MODEL_ID, device="cpu", compute_type="int8")

HINGLISH_PROMPT = "This is a mixed Hindi and English conversation. Keep Hindi words in Latin script. Do not translate. Example: Mera laptop kharab ho gaya."

class DraftState:
    def __init__(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.committed_text = ""
        self.lock = threading.Lock()

state = DraftState()

def draft(audio_chunk: np.ndarray, is_final: bool = False) -> dict:
    with state.lock:
        state.audio_buffer = np.concatenate((state.audio_buffer, audio_chunk))
        
        if len(state.audio_buffer) < 16000 * 0.5 and not is_final:
            return {"partial": "", "committed": state.committed_text}

        segments, info = model.transcribe(
            state.audio_buffer,
            language="hi",
            initial_prompt=HINGLISH_PROMPT,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False
        )
        
        current_text = "".join(segment.text for segment in segments).strip()

        if is_final:
            state.committed_text += " " + current_text
            state.audio_buffer = np.array([], dtype=np.float32)
            return {"partial": "", "committed": state.committed_text.strip()}

        return {"partial": current_text, "committed": state.committed_text.strip()}
