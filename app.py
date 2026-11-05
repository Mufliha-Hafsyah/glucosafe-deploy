import os
import json
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template_string

app = Flask(__name__)

MODEL_PATH = 'glucosafe_model.tflite'
SCALER_PATH = 'glucosafe_scaler.pkl'
FEATURES_PATH = 'glucosafe_features.json'

# Load Model TFLite, Scaler, dan Urutan Fitur
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
        
    with open(FEATURES_PATH, 'r') as f:
        feature_names = json.load(f)
        
    print("✅ System Loaded Successfully (GlucoSafe TFLite)")
except Exception as e:
    print(f"❌ Error loading system: {e}")

# Template HTML dengan gaya GlucoSafe (Hijau Tosca)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GlucoSafe - Skrining Diabetes</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #F4F7F6; color: #333; margin: 0; padding: 20px; }
        .container { max-width: 600px; margin: auto; background: white; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); overflow: hidden; }
        .header { background-color: #0D9488; color: white; padding: 20px; text-align: center; }
        .header h1 { margin: 0; font-size: 24px; }
        .header p { margin: 5px 0 0; font-size: 14px; opacity: 0.9; }
        .form-group { padding: 15px 20px; border-bottom: 1px solid #eee; }
        .form-group label { display: block; font-weight: 600; margin-bottom: 8px; color: #0D9488; }
        input[type="number"], select { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px; box-sizing: border-box; }
        .radio-group { display: flex; gap: 15px; }
        .radio-group label { font-weight: normal; color: #333; display: flex; align-items: center; gap: 5px; cursor: pointer;}
        .btn-submit { background-color: #0D9488; color: white; border: none; padding: 15px; width: 100%; font-size: 16px; font-weight: bold; border-radius: 8px; cursor: pointer; margin-top: 20px; transition: 0.3s; }
        .btn-submit:hover { background-color: #0f766e; }
        .result { text-align: center; padding: 40px 20px; }
        .result h2 { color: #0D9488; }
        .score { font-size: 48px; font-weight: bold; color: #0D9488; margin: 20px 0; }
        .btn-back { display: inline-block; margin-top: 20px; text-decoration: none; color: white; background-color: #0D9488; padding: 10px 20px; border-radius: 8px; }
    </style>
</head>
<body>
    <div class="container">
        {% if result %}
            <div class="header"><h1>Hasil Skrining</h1></div>
            <div class="result">
                <h2>Risiko Diabetes Anda:</h2>
                <div class="score">{{ result }}</div>
                <p>{% if color == 'red' %}⚠️ Segera konsultasikan ke dokter.{% else %}✅ Tetap jaga pola hidup sehat!{% endif %}</p>
                <a href="/" class="btn-back">Kembali ke Form</a>
            </div>
        {% else %}
            <div class="header">
                <h1>GlucoSafe</h1>
                <p>Skrining Risiko Diabetes</p>
            </div>
            <form action="/predict" method="post">
                <div class="form-group">
                    <label>Data Diri (Langkah 1)</label>
                    <label style="color:#333; font-weight:normal;">Jenis Kelamin (0=Wanita, 1=Pria)</label>
                    <select name="Sex"><option value="0">Wanita</option><option value="1">Pria</option></select><br><br>
                    <label style="color:#333; font-weight:normal;">Umur (Kategori 1-13)</label>
                    <input type="number" name="Age" min="1" max="13" value="5" required><br><br>
                    <label style="color:#333; font-weight:normal;">BMI (Indeks Massa Tubuh)</label>
                    <input type="number" step="0.1" name="BMI" placeholder="Misal: 24.5" required>
                </div>
                
                <div class="form-group">
                    <label>Riwayat Diagnosis (Langkah 2)</label>
                    <label style="color:#333; font-weight:normal;">Tekanan Darah Tinggi</label>
                    <div class="radio-group"><label><input type="radio" name="HighBP" value="1" required> Ya</label><label><input type="radio" name="HighBP" value="0"> Tidak</label></div><br>
                    <label style="color:#333; font-weight:normal;">Kolesterol Tinggi</label>
                    <div class="radio-group"><label><input type="radio" name="HighChol" value="1" required> Ya</label><label><input type="radio" name="HighChol" value="0"> Tidak</label></div><br>
                    <label style="color:#333; font-weight:normal;">Stroke</label>
                    <div class="radio-group"><label><input type="radio" name="Stroke" value="1" required> Ya</label><label><input type="radio" name="Stroke" value="0"> Tidak</label></div><br>
                    <label style="color:#333; font-weight:normal;">Penyakit Jantung</label>
                    <div class="radio-group"><label><input type="radio" name="HeartDiseaseorAttack" value="1" required> Ya</label><label><input type="radio" name="HeartDiseaseorAttack" value="0"> Tidak</label></div><br>
                    <label style="color:#333; font-weight:normal;">Kesulitan Berjalan</label>
                    <div class="radio-group"><label><input type="radio" name="DiffWalk" value="1" required> Ya</label><label><input type="radio" name="DiffWalk" value="0"> Tidak</label></div>
                </div>

                <div class="form-group">
                    <label>Gaya Hidup (Langkah 3)</label>
                    <label style="color:#333; font-weight:normal;">Merokok</label>
                    <div class="radio-group"><label><input type="radio" name="Smoker" value="1" required> Ya</label><label><input type="radio" name="Smoker" value="0"> Tidak</label></div><br>
                    <label style="color:#333; font-weight:normal;">Aktivitas Fisik 30 Hari Terakhir</label>
                    <div class="radio-group"><label><input type="radio" name="PhysActivity" value="1" required> Ya</label><label><input type="radio" name="PhysActivity" value="0"> Tidak</label></div><br>
                    <label style="color:#333; font-weight:normal;">Konsumsi Buah</label>
                    <div class="radio-group"><label><input type="radio" name="Fruits" value="1" required> Ya</label><label><input type="radio" name="Fruits" value="0"> Tidak</label></div><br>
                    <label style="color:#333; font-weight:normal;">Konsumsi Sayur</label>
                    <div class="radio-group"><label><input type="radio" name="Veggies" value="1" required> Ya</label><label><input type="radio" name="Veggies" value="0"> Tidak</label></div>
                </div>

                <div class="form-group">
                    <label>Kondisi Umum (Langkah 4)</label>
                    <label style="color:#333; font-weight:normal;">Skor Kesehatan Umum (1=Sangat Buruk, 5=Sangat Baik)</label>
                    <input type="number" name="GenHlth" min="1" max="5" value="3" required><br><br>
                    <label style="color:#333; font-weight:normal;">Hari Kesehatan Mental Buruk (0-30)</label>
                    <input type="number" name="MentHlth" min="0" max="30" value="0" required><br><br>
                    <label style="color:#333; font-weight:normal;">Hari Kesehatan Fisik Buruk (0-30)</label>
                    <input type="number" name="PhysHlth" min="0" max="30" value="0" required>
                </div>
                
                <div style="padding: 20px;">
                    <button type="submit" class="btn-submit">Lihat Hasil Skrining</button>
                </div>
            </form>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari form berdasarkan nama field
        form_data = request.form
        
        # Susun array sesuai urutan fitur yang disimpan di json
        input_data = []
        for feature in feature_names:
            val = float(form_data[feature])
            input_data.append(val)
            
        # Ubah ke numpy array dan lakukan scaling
        input_array = np.array([input_data], dtype=np.float32)
        scaled_data = scaler.transform(input_array).astype(np.float32)
        
        # Prediksi dengan TFLite
        interpreter.set_tensor(input_details[0]['index'], scaled_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
        
        # Hitung persentase
        prob_percent = round(prediction * 100, 1)
        
        if prediction > 0.5:
            result_text = f"{prob_percent}% (Risiko Tinggi)"
            color = "red"
        else:
            result_text = f"{round(100 - prob_percent, 1)}% (Risiko Rendah)"
            color = "green"
            
        return render_template_string(HTML_TEMPLATE, result=result_text, color=color)
    except Exception as e:
        return f"Error: {e}"
        
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)