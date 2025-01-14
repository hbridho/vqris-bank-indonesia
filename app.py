from flask import Flask, render_template, request, jsonify, send_from_directory
from transformers import VitsModel, AutoTokenizer
import torch
import scipy.io.wavfile
import os
import re
import logging
from flask_cors import CORS

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app) 

logging.basicConfig(level=logging.DEBUG)

model_path = "model/mms-tts-ind"
try:
    model = VitsModel.from_pretrained(model_path)
    app.logger.info("Model loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading model: {e}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    app.logger.info("Tokenizer loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading tokenizer: {e}")

def num_to_text(num):
    satuan = ["", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan", "sepuluh", "sebelas"]
    def terbilang(n):
        if n < 12:
            return satuan[n]
        elif n < 20:
            return satuan[n - 10] + " belas"
        elif n < 100:
            return satuan[n // 10] + " puluh " + satuan[n % 10]
        elif n < 200:
            return "seratus " + terbilang(n - 100)
        elif n < 1000:
            return satuan[n // 100] + " ratus " + terbilang(n % 100)
        elif n < 2000:
            return "seribu " + terbilang(n - 1000)
        elif n < 1000000:
            return terbilang(n // 1000) + " ribu " + terbilang(n % 1000)
        elif n < 1000000000:
            return terbilang(n // 1000000) + " juta " + terbilang(n % 1000000)
        else:
            return "Angka terlalu besar"
    num = int(num.replace(",", "").replace(".", ""))
    return terbilang(num).strip()

def strip_number(text):
    stripped_text = re.sub(r'[^\d,\.]', '', text)
    if stripped_text:
        parts = stripped_text.split(',')
        if len(parts) > 1:
            main_part = parts[0].replace('.', '')
            decimal_part = parts[1]
            return f"{int(main_part):,}.{decimal_part}".replace(',', '.')
        else:
            return f"{int(stripped_text.replace('.', '')):,}".replace(',', '.')
    return stripped_text

def generate_audio(input_text, output_filename):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform.squeeze().cpu().numpy()
    output_path = os.path.join(output_dir, output_filename)
    scipy.io.wavfile.write(output_path, rate=model.config.sampling_rate, data=output)
    return output_filename

def tts_acknowledge(input_text):
    shown_number = strip_number(input_text)
    text_for_conversion = shown_number.replace(".", "").replace(",", "")
    input_text = num_to_text(text_for_conversion)
    return shown_number, generate_audio(f"Anda akan membuat pembayaran senilai {input_text} rupiah, Apakah benar?", "acknowledge.wav")

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/speech_to_text', methods=['POST'])
def speech_to_text():
    data = request.get_json()
    text = data['text']
    
    if text.lower() in ["iya", "yes"]:
        return jsonify({"response": "proceed"})
    
    shown_number, audio_filename = tts_acknowledge(text)
    response = {'audio': audio_filename, 'shown_number': shown_number}
    return jsonify(response)

@app.route('/tts_suggest', methods=['GET'])
def get_tts_suggest():
    return send_from_directory('static/voices', 'suggesting.wav')

@app.route('/tts_input_pin', methods=['GET'])
def get_tts_input_pin():
    return send_from_directory('static/voices', 'input_pin.wav')

@app.route('/tts_input_pin_success', methods=['GET'])
def get_tts_input_pin_success():
    return send_from_directory('static/voices', 'input_pin_success.wav')

@app.route('/tts_input_pin_failed', methods=['GET'])
def get_tts_input_pin_failed():
    return send_from_directory('static/voices', 'input_pin_failed.wav')

@app.route('/tts_asking', methods=['GET'])
def get_tts_asking():
    return send_from_directory('static/voices', 'asking.wav')

@app.route('/tts_failed', methods=['GET'])
def get_tts_failed():
    return send_from_directory('static/voices', 'failed.wav')

@app.route('/tts_confirmation', methods=['GET'])
def get_tts_confirmation():
    return send_from_directory('static/voices', 'confirmation.wav')

@app.route('/audio/<filename>')
def get_audio(filename):
    return send_from_directory('output', filename)

@app.route('/static/voices/start_rec.wav')
def get_cue_sound():
    return send_from_directory('static/voices', 'start_rec.wav')

@app.route('/validate_pin', methods=['POST'])
def validate_pin():
    data = request.get_json()
    pin = data.get('pin', '')
    correct_pin = '1234'
    if pin == correct_pin:
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'failure'})

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
