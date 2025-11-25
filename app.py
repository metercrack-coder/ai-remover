from flask import Flask, request, send_file, jsonify, render_template
from flask_cors import CORS
from rembg import remove, new_session
from PIL import Image
import io
import base64

# Flask app
app = Flask(__name__, template_folder="templates")  # Added template folder for index.html
CORS(app, resources={r"/*": {"origins": "*"}})      # Allow all origins for Render

# Initialize different AI models for various use cases
sessions = {
    'u2net': new_session('u2net'),
    'u2netp': new_session('u2netp'),
    'u2net_human_seg': new_session('u2net_human_seg'),
}

# Serve index.html from templates folder
@app.route('/')
def home():
    return render_template('index.html')

# Remove background endpoint
@app.route('/remove-background', methods=['POST'])
def remove_background():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        model = request.form.get('model', 'u2net')
        if model not in sessions:
            model = 'u2net'

        input_image = Image.open(file.stream)

        # Convert to RGB if needed
        if input_image.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', input_image.size, (255, 255, 255))
            if input_image.mode == 'P':
                input_image = input_image.convert('RGBA')
            background.paste(input_image, mask=input_image.split()[-1] if input_image.mode in ('RGBA', 'LA') else None)
            input_image = background
        elif input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')

        output_image = remove(
            input_image,
            session=sessions[model],
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10
        )

        img_io = io.BytesIO()
        output_image.save(img_io, 'PNG', optimize=True)
        img_io.seek(0)
        img_base64 = base64.b64encode(img_io.getvalue()).decode()

        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{img_base64}',
            'model_used': model
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Models endpoint
@app.route('/models', methods=['GET'])
def get_models():
    return jsonify({
        'models': [
            {'id': 'u2net', 'name': 'U2-Net (General)', 'description': 'Best for general use - objects, people, animals'},
            {'id': 'u2netp', 'name': 'U2-Net+ (Fast)', 'description': 'Faster processing, good quality'},
            {'id': 'u2net_human_seg', 'name': 'Human Segmentation', 'description': 'Optimized for removing backgrounds from photos of people'}
        ]
    })

# Health check
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

# Production-ready run
if __name__ == '__main__':
    print("🚀 AI Background Remover Server Starting...")
    print("📦 Loading AI models (this may take a moment on first run)...")
    print("✅ Server ready")
    app.run(host="0.0.0.0", port=5000)  # Listen on all interfaces for Render
