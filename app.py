import gradio as gr
from transformers import pipeline

# Use minimal models known to work on CPU Spaces
stt = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
text_gen = pipeline("text-generation", model="sshleifer/tiny-gpt2")

def ai_sloka_tutor(audio):
    try:
        # Step 1: Transcribe audio
        result = stt(audio)
        sloka_text = result.get("text", "").strip()

        if not sloka_text:
            return "⚠️ No recognizable speech. Please try again."

        # Step 2: Generate simplified meaning
        prompt = f"Translate and explain this Sanskrit sloka: {sloka_text}"
        ai_output = text_gen(prompt, max_length=50, num_return_sequences=1)[0]["generated_text"]
        meaning = ai_output.replace(prompt, "").strip()

        return f"🎧 Sloka: {sloka_text}\n\n🧠 AI Meaning: {meaning or 'No meaning generated.'}"

    except Exception as error:
        return f"❌ Runtime error: {str(error)}"

# Gradio Interface
interface = gr.Interface(
    fn=ai_sloka_tutor,
    inputs=gr.Audio(source="upload", type="filepath", label="Upload WAV file"),
    outputs="text",
    title="AI Sloka Tutor",
    description="Upload a Sanskrit sloka audio file to receive a simplified English explanation."
)

interface.launch()
