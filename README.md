# Cookify: Recipe Extraction from Cooking Videos

Cookify is a Python-based tool that extracts structured recipe information from cooking videos. It uses computer vision, speech recognition, and natural language processing to identify ingredients, tools, cooking steps, and other recipe components.

## ✨ Recent Improvements (cai_improvements branch)

- **Fixed directory structure**: Eliminated nested `cookify` directory
- **Enhanced error handling**: Comprehensive error handling throughout the pipeline
- **Improved model loading**: Robust model loading with fallback strategies
- **Better dependency management**: Updated dependencies for Python 3.12 compatibility
- **Working examples**: Added functional example scripts
- **Graceful degradation**: System continues working even if optional dependencies are missing

## Features

- Extract recipe title, servings, and ingredients with quantities and units
- Identify cooking tools used in the video
- Extract step-by-step cooking instructions with timestamps
- Recognize cooking actions and techniques
- Generate structured JSON output of the complete recipe
- **Robust error handling** and graceful degradation
- **Multiple model fallback strategies** for better reliability

## System Requirements

- Python 3.8 or higher (Python 3.12 supported)
- FFmpeg (for video and audio processing)
- CUDA-compatible GPU recommended for faster processing (but not required)

## Quick Start

```bash
# Clone and navigate to the project
git clone <repository-url>
cd cookify

# Install dependencies and run tests
python install_and_test.py

# Run the working example
python examples/working_example.py
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/kapeleshh/cookify.git
cd cookify
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install the package and dependencies

```bash
pip install -e .
```

This will install all required dependencies including:
- OpenCV and FFmpeg for video processing
- PyTorch and YOLOv8 for object detection
- EasyOCR for text recognition
- Whisper for speech-to-text
- spaCy for NLP

### 4. Download pre-trained models

```bash
python -m cookify.src.utils.model_downloader
```

## Usage

### Basic Usage

```bash
cookify path/to/cooking/video.mp4
```

This will process the video and save the extracted recipe as `recipe.json` in the current directory.

### Advanced Options

```bash
cookify path/to/cooking/video.mp4 --output custom_output.json --verbose
```

For more options:

```bash
cookify --help
```

## Web Interface

Cookify includes a modern web interface for easy video upload and recipe extraction.

### Starting the Web Server

1. **Activate the virtual environment** (if not already active):
   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Start the Flask web server**:
   ```bash
   python src/ui/app.py
   ```

3. **Open your web browser** and navigate to:
   - **http://127.0.0.1:5000** (localhost)
   - **http://0.0.0.0:5000** (network access)

### Using the Web Interface

1. **Upload a Video**: 
   - Drag and drop a cooking video onto the upload area, or
   - Click "Browse Files" to select a video file
   - Supported formats: MP4, AVI, MOV, MKV, WEBM (max 50MB)

2. **Processing**: 
   - The system will automatically process your video
   - A progress indicator shows the processing status
   - Processing time depends on video length (typically 1-5 minutes)

3. **View Results**: 
   - Once complete, click "View Recipe" to see the extracted recipe
   - The results page displays:
     - Recipe title and metadata (servings, total time)
     - Ingredients list with quantities and units
     - Step-by-step cooking instructions
     - Cooking tools used
     - Interactive video player with clickable timestamps for each step

4. **Download Results**: 
   - Click "Download Recipe (JSON)" to save the structured recipe data
   - Use "Upload Another Video" to process additional videos

### Web Interface Features

- **Modern UI**: Clean, responsive design built with Bootstrap 5
- **Drag & Drop Upload**: Easy video file upload with visual feedback
- **Real-time Progress**: Live progress updates during video processing
- **Interactive Results**: Click on cooking steps to jump to specific video timestamps
- **Mobile Friendly**: Responsive design works on desktop and mobile devices
- **Error Handling**: Clear error messages and graceful failure handling

### Troubleshooting Web Interface

If you encounter issues starting the web server:

1. **Check dependencies**: Ensure Flask is installed:
   ```bash
   pip install flask
   ```

2. **Verify Python path**: Make sure you're running from the project root directory

3. **Check port availability**: If port 5000 is in use, the server will show an error. You can modify the port in `src/ui/app.py`:
   ```python
   app.run(debug=True, host='0.0.0.0', port=5001)  # Change port number
   ```

4. **View logs**: The terminal will show detailed logs of any errors during startup or processing

## Output Format

The extracted recipe is saved as a JSON file with the following structure:

```json
{
  "title": "Recipe Title",
  "servings": "Number of servings",
  "ingredients": [
    {"name": "ingredient name", "qty": "quantity", "unit": "measurement unit"}
  ],
  "tools": ["tool1", "tool2"],
  "steps": [
    {
      "idx": "step number",
      "start": "timestamp start",
      "end": "timestamp end",
      "action": "cooking action",
      "objects": ["ingredients/tools involved"],
      "details": "additional instructions",
      "temp": "temperature (optional)",
      "duration": "cooking duration (optional)"
    }
  ]
}
```

## Project Structure

```
cookify/
├── data/               # Data directory for input/output files
│   ├── input/          # Input videos
│   └── output/         # Output recipes and processed data
├── models/             # Pre-trained models
├── src/                # Source code
│   ├── preprocessing/  # Video preprocessing
│   ├── frame_analysis/ # Frame analysis (object detection, OCR)
│   ├── audio_analysis/ # Audio transcription and NLP
│   ├── integration/    # Multimodal integration
│   ├── recipe_extraction/ # Recipe structure extraction
│   ├── output_formatting/ # Output formatting
│   └── utils/          # Utility functions
├── tests/              # Unit tests
├── main.py             # Main entry point
├── requirements.txt    # Dependencies
└── setup.py            # Setup script
```

## Development

### Running Tests

```bash
pytest tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

[MIT License](LICENSE)
