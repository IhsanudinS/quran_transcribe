from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pydub import AudioSegment
import io
import logging
import numpy as np
from difflib import SequenceMatcher
import re

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG)

processor = Wav2Vec2Processor.from_pretrained("sankun/wav2vec2-quran")
model = Wav2Vec2ForCTC.from_pretrained("sankun/wav2vec2-quran")

# Fungsi untuk menghapus tanda-tanda waqof dan tanda ayat sajdah
def normalize_text(text):
    waqof_signs = ["ۖ", "ۗ", "ۚ", "ۛ", "ۜ", "۩", "ْ"]
    for sign in waqof_signs:
        text = text.replace(sign, "")
    # Menghapus spasi kosong yang berlebihan
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Fungsi untuk mengecek apakah karakter adalah harakat
def is_harakat(char):
    harakats = "ًٌٍَُِّْ"
    return char in harakats

def compare_texts(original, transcription):
    original_normalized = normalize_text(original)
    transcription_normalized = normalize_text(transcription)
    
    matcher = SequenceMatcher(None, original_normalized, transcription_normalized)
    differences = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            error_type = 'Unknown'
            if tag == 'replace':
                if len(original_normalized[i1:i2]) == len(transcription_normalized[j1:j2]):
                    # Check if the replacement is harakat or letter
                    if all(is_harakat(c) for c in original_normalized[i1:i2] + transcription_normalized[j1:j2]):
                        error_type = 'Kesalahan Harakat'
                    else:
                        error_type = 'Kesalahan Huruf'
                else:
                    error_type = 'Kesalahan Huruf'
            elif tag == 'delete':
                error_type = 'Kata yang Hilang'
            elif tag == 'insert':
                error_type = 'Kata Tambahan'

            differences.append({
                'type': error_type,
                'original': original_normalized[i1:i2],
                'transcription': transcription_normalized[j1:j2]
            })

    correct_chars = sum(len(original_normalized[i1:i2]) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag == 'equal')
    total_chars = len(original_normalized)
    accuracy = (correct_chars / total_chars) * 100

    return differences, accuracy

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    original_text = request.form['original_text']

    try:
        audio = AudioSegment.from_file(io.BytesIO(file.read()), format="wav")
        audio = audio.set_frame_rate(16000).set_channels(1)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

        input_values = processor(samples, sampling_rate=16000, return_tensors="pt").input_values
        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])

        differences, accuracy = compare_texts(original_text, transcription)

        return jsonify({"transcription": transcription, "differences": differences, "accuracy": accuracy})

    except Exception as e:
        logging.error("Error during transcription: %s", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')