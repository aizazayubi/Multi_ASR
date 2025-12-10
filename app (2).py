import gradio as gr
from transformers import pipeline

# ----------------- LOAD MODELS -----------------
pashto_asr = pipeline(
    "automatic-speech-recognition",
    model="https://huggingface.co/models/Aizazayyubi/pashto-whisper-asr",
    device=-1
)

khowar_asr = pipeline(
    "automatic-speech-recognition",
    model="https://huggingface.co/models/Aizazayyubi/khowar-whisper-asr",
    device=-1
)

def transcribe(audio, lang_choice):
    if audio is None:
        return "No audio provided"

    if lang_choice == "Pashto":
        return pashto_asr(audio)["text"]
    else:
        return khowar_asr(audio)["text"]

# ----------------- CSS -----------------
custom_css = """
body, .gradio-container { background: #f2f4f7 !important; }

/* Card */
.card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.08);
    margin-bottom: 18px;
}

/* Header - SOLID BLACK */
.header-title {
    text-align: center;
    font-size: 36px;
    font-weight: 800;
    color: white;
    padding: 20px;
    background: #000;
    border-radius: 12px;
    margin-bottom: 4px;
}

.header-sub {
    text-align: center;
    font-size: 18px;
    color: white;
    padding: 10px;
    background: #000;
    border-radius: 8px;
    margin-bottom: 25px;
}

/* Developer Section - SOLID BLACK */
.dev-box {
    text-align: center;
    background: #000;
    padding: 22px;
    border-radius: 16px;
    margin-top: 25px;
    color: white;
}

.dev-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 8px;
    color: white;
}

.dev-text {
    font-size: 15px;
    opacity: 0.95;
    color: white;
}

footer { visibility: hidden !important; }
"""

# ----------------- UI -----------------
with gr.Blocks() as iface:

    gr.HTML(f"<style>{custom_css}</style>")

    gr.HTML("""
    <h1 class='header-title'>Multilingual Speech-to-Text</h1>
    <p class='header-sub'>Pashto & Khowar | Whisper-Based ASR</p>
    """)

    with gr.Row(elem_classes="card"):
        with gr.Column():
            lang_choice = gr.Radio(
                ["Pashto", "Khowar"],
                value="Pashto",
                label="Select Language"
            )

            audio_in = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="🎤 Upload or Record Audio"
            )

            btn = gr.Button("Transcribe")

        with gr.Column():
            output = gr.Textbox(
                label="Transcription Output",
                lines=10
            )

    btn.click(transcribe, [audio_in, lang_choice], output)

    # Developer Section
    gr.HTML("""
    <div class="dev-box">
        <div class="dev-title">Developed by The Speech Rangers</div>
        <div class="dev-text">
            Building speech technology for Pakistan’s low-resource languages.
        </div>
    </div>
    """)

iface.launch()
