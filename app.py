from flask import Flask, render_template, request, jsonify, send_from_directory
from transformers import VitsModel, AutoTokenizer
import torch
import scipy.io.wavfile
import os
import re

app = Flask(__name__, template_folder="templates")

model_path = "model/mms-tts-ind"
model = VitsModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

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

def clear_output_folder(output_dir):
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)

def tts_inference(input_text):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    clear_output_folder(output_dir)

    shown_number = strip_number(input_text)
    text_for_conversion = shown_number.replace(".", "").replace(",", "")
    input_text = num_to_text(text_for_conversion)
    input_text = f"anda akan membayar dengan fikris sebesar {input_text} rupiah"
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform.squeeze().cpu().numpy()
    
    output_path = os.path.join(output_dir, "output.wav")
    scipy.io.wavfile.write(output_path, rate=model.config.sampling_rate, data=output)
    return shown_number, "output.wav"

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/speech_to_text', methods=['POST'])
def speech_to_text():
    data = request.get_json()
    text = data['text']
    shown_number, audio_filename = tts_inference(text)
    response = {'audio': audio_filename, 'shown_number': shown_number}
    return jsonify(response)

@app.route('/audio/<filename>')
def get_audio(filename):
    return send_from_directory('output', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)