import gradio as gr
from transformers import pipeline

# ----------------- LOAD MODELS -----------------
pashto_asr = pipeline(
    "automatic-speech-recognition",
    model="Aizazayyubi/pashto-whisper-asr",
    device=-1
)

khowar_asr = pipeline(
    "automatic-speech-recognition",
    model="Aizazayyubi/khowar-whisper-asr",
    device=-1
)

# ----------------- TRANSCRIPTION -----------------
def transcribe(audio, lang):
    if audio is None:
        return "❌ No audio provided."

    if lang == "Pashto":
        return pashto_asr(audio)["text"]
    return khowar_asr(audio)["text"]

# ----------------- CUSTOM CSS -----------------
custom_css = """
body, .gradio-container {
    background: linear-gradient(135deg, #eef2f7, #f9fafb) !important;
}

.app-card {
    background: white;
    padding: 28px;
    border-radius: 22px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.08);
}

.header {
    text-align: center;
    padding: 28px;
    background: #000;
    color: white;
    border-radius: 18px;
    margin-bottom: 25px;
}

.header h1 {
    font-size: 38px;
    font-weight: 900;
    margin-bottom: 8px;
}

.header p {
    font-size: 18px;
    opacity: 0.9;
}

.lang-badge {
    background: #000;
    color: white;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 600;
}

.dev-box {
    text-align: center;
    background: #000;
    padding: 24px;
    border-radius: 18px;
    margin-top: 30px;
    color: white;
}

footer { display: none !important; }
"""

# ----------------- UI -----------------
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as iface:

    # Header
    gr.HTML("""
    <div class="header">
        <h1>Multilingual Speech-to-Text</h1>
        <p>Whisper-based ASR for Pashto & Khowar</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1, elem_classes="app-card"):
            gr.Markdown("### 🌐 Language Selection")
            lang = gr.Radio(
                ["Pashto", "Khowar"],
                value="Pashto",
                interactive=True
            )

            gr.Markdown("### 🎧 Audio Input")
            with gr.Tabs():
                with gr.Tab("🎙️ Record"):
                    audio = gr.Audio(
                        sources=["microphone"],
                        type="filepath"
                    )
                with gr.Tab("📁 Upload"):
                    audio = gr.Audio(
                        sources=["upload"],
                        type="filepath"
                    )

            transcribe_btn = gr.Button("🚀 Transcribe", variant="primary")

        with gr.Column(scale=1, elem_classes="app-card"):
            gr.Markdown("### 📝 Transcription Output")
            output = gr.Textbox(
                lines=12,
                placeholder="Your transcription will appear here..."
            )

            with gr.Row():
                gr.Button("📋 Copy", onclick="navigator.clipboard.writeText(document.querySelector('textarea').value)")
                gr.Button("🧹 Clear", onclick="document.querySelector('textarea').value=''")

    transcribe_btn.click(
        fn=transcribe,
        inputs=[audio, lang],
        outputs=output
    )

    # Developer Section
    gr.HTML("""
    <div class="dev-box">
        <h3>Developed by The Speech Rangers</h3>
        <p>Empowering Pakistan’s low-resource languages with AI-driven speech technology.</p>
    </div>
    """)

iface.launch()
