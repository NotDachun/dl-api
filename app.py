import sys
import generate_text
import generate_visuals
import numpy as np
from models import ScriptGenModelNLayer
from vocab import Vocabulary
from flask import Flask, request, jsonify
from pathlib import Path

sys.path.append("./Real-Time-Voice-Cloning")
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder

app = Flask(__name__)

DEVICE = "cpu"
SEQUENCE_LENGTH = 512
TEMPERATURE = 0.5
BEAM_WIDTH = 10

# Load pre-determined vocabs
office_vocab = Vocabulary("./static/vocabs/office_transcript_end_scene_chars_test.pkl")
casey_vocab = Vocabulary("./static/vocabs/cleaned_casey_larger_chars_test.pkl")
documentary_vocab = Vocabulary("./static/vocabs/nova_ALL_transcripts_chars_test.pkl")
nature_vocab = Vocabulary("./static/vocabs/nova_nature_transcripts_chars_test.pkl")
mark_vocab = Vocabulary("./static/vocabs/cleaned_mark_rober_linebreak_chars_test.pkl")

vocabs = {
    "office": office_vocab,
    "casey": casey_vocab,
    "documentary": documentary_vocab,
    "nature": nature_vocab,
    "mark": mark_vocab
}

# Load pre-trained models
office_model = ScriptGenModelNLayer(len(office_vocab), 512, 2)
office_model.load_last_model("./static/models/3.1.office/checkpoints")
casey_model = ScriptGenModelNLayer(len(casey_vocab), 512, 2)
casey_model.load_last_model("./static/models/5.1.Casey/checkpoints")
documentary_model = ScriptGenModelNLayer(len(documentary_vocab), 512, 2)
documentary_model.load_last_model("./static/models/6.1.Documentary/checkpoints")
nature_model = ScriptGenModelNLayer(len(nature_vocab), 512, 2)
nature_model.load_last_model("./static/models/7.1.DocumentaryNature/checkpoints")
mark_model = ScriptGenModelNLayer(len(mark_vocab), 512, 2)
mark_model.load_last_model("./static/models/8.1.MarkRober/checkpoints")

models = {
    "office": office_model,
    "casey": casey_model,
    "documentary": documentary_model,
    "nature": nature_model,
    "mark": mark_model
}

# Cells to visualize for each model
vis_cells = {
    "office": [700, 652, 625, 643]
}

# Load pretrained voice cloning models
# encoder_weights = Path("./Real-Time-Voice-Cloning/encoder/saved_models/pretrained.pt")
vocoder_weights = Path("./Real-Time-Voice-Cloning/vocoder/saved_models/pretrained/pretrained.pt")
syn_dir = Path("./Real-Time-Voice-Cloning/synthesizer/saved_models/logs-pretrained/taco_pretrained")
# encoder.load_model(encoder_weights)
synthesizer = Synthesizer(syn_dir)
vocoder.load_model(vocoder_weights)

# Load pretrained voice embeddings
stuart_embeddings = np.load("./static/voice/stuart_embed.npy")
embeds = {
    "stuart": stuart_embeddings
}

@app.route('/generateText', methods=["POST"])
def gen_text():
    """
    Accepts POST request with payload like:
        {"seedWords": text, "model": model}
    Then, returns a payload:
        {"text": generated text}
    """
    if request.method == "POST":
        print(request.form, file=sys.stderr)
        
        model_name = request.form["model"]
        seed_words = request.form["seedWords"]

        model = models[model_name]
        vocab = vocabs[model_name]

        output = generate_text.generate_language(model, vocab, seed_words, SEQUENCE_LENGTH, DEVICE, sampling_strategy="sample")
        return jsonify({'text': output})
    return "Make POST request to generate text using a pre-trained language model"

@app.route('/generateCellVis', methods=["POST"])
def gen_vis():
    """
    Accepts POST request with payload like:
        {"text": text, "model": model}
    Then, returns a payload:
        {"cell_vis": visualization}
    """
    if request.method == "POST":
        print(request.form, file=sys.stderr)

        text = request.form["text"]
        model_name = request.form["model"]

        model = models[model_name]
        vocab = vocabs[model_name]
        cells = vis_cells.get(model_name, None)

        output = generate_visuals.generate_cell_visualization(model, vocab, text, DEVICE, cells)
        return jsonify({'cell_vis': output})
    return "Make POST request to generate visualization using a pre-trained language model"

@app.route('/generateAudio', methods=["POST"])
def gen_audio():
    """
    Accepts POST request with payload like:
        {"text": text, "embed": embed}
    Then, returns a file:
        {"audio": audio_file}
    """    
    if request.method == "POST":
        print(request.form, file=sys.stderr)

        text = request.form["text"]
        embed_name = request.form["embed"]

        embed = embeds[embed_name]

        output = generate_visuals.generate_cell_visualization(model, vocab, text, DEVICE, cells)
        return jsonify({'cell_vis': output})
    return "Make POST request to generate visualization using a pre-trained language model"


