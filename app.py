import streamlit as st
import torchaudio
from transformers import pipeline

st.set_page_config(page_title="AI Sloka Tutor", layout="centered")

st.title("🧘‍♀️ AI Sloka Tutor (Voice to Meaning)")
st.markdown("Upload a Sanskrit sloka audio file (.wav) to get a simplified English explanation.")

@st.cache_resource
def load_models():
    stt = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
    nlp = pipeline("text-generation", model="sshleifer/tiny-gpt2")
    return stt, nlp

stt, nlp = load_models()

audio_file = st.file_uploader("🎙️ Upload a .wav file", type=["wav"])

if audio_file:
    st.audio(audio_file)

    with st.spinner("Transcribing and interpreting..."):
        try:
            # Read file using torchaudio
            waveform, sample_rate = torchaudio.load(audio_file)
            waveform = waveform.squeeze().numpy()

            result = stt(waveform)
            sloka_text = result.get("text", "").strip()

            st.subheader("📜 Transcription")
            st.success(sloka_text)

            prompt = f"Explain this Sanskrit sloka in simple English: {sloka_text}"
            generated = nlp(prompt, max_length=60, do_sample=True)[0]["generated_text"]
            meaning = generated.replace(prompt, "").strip()

            st.subheader("🧠 Explanation")
            st.info(meaning or "AI could not generate an explanation.")

        except Exception as e:
            st.error(f"❌ Error: {e}")

           
