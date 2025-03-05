import os
import subprocess
import uuid
from flask import Flask, request, jsonify, send_file
from pyngrok import ngrok
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

public_url = ngrok.connect(5000)
print(f"Flask app is live at {public_url}")

output_dir = './results'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/run_inference', methods=['POST'])
def run_inference():
    if 'condition_image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    if len(request.files.getlist('condition_image')) > 1:
        return jsonify({'error': 'Only one image can be uploaded at a time'}), 400
    
    image_file = request.files['condition_image']
    
    if not allowed_file(image_file.filename):
        return jsonify({'error': 'Invalid image format. Allowed formats are PNG, JPG, JPEG.'}), 400
    
    unique_id = str(uuid.uuid4())
    image_path = f"/content/UW-DiffPhys_webapp/{unique_id}_{image_file.filename}"
    image_file.save(image_path)
    
    command = [
        "python", "./inference_UW-DDIM.py",
        "--config", "./underwater_lsui_uieb_128.yml",
        "--resume", "/content/Checkpoints",
        "--sampling_timesteps", "25",
        "--eta", "0",
        "--condition_image", image_path,
        "--seed", "5"
    ]
    
    try:
        subprocess.run(command, check=True)
        
        result_file = os.path.join(output_dir, f"{unique_id}_{os.path.splitext(image_file.filename)[0]}.png")
        
        if os.path.exists(result_file):
            response = send_file(result_file, mimetype='image/png')
            
            if os.path.exists(image_path):
                os.remove(image_path)
            
            return response
        else:
            return jsonify({'error': 'Result file not found'}), 500
    
    except subprocess.CalledProcessError as e:
        return jsonify({'error': f'Error running inference: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(port=5000)
