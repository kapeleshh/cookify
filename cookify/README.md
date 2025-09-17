# Cookify: Recipe Extraction from Cooking Videos

Cookify is a Python-based tool that extracts structured recipe information from cooking videos. It uses computer vision, speech recognition, and natural language processing to identify ingredients, tools, cooking steps, and other recipe components.

## Features

- Extract recipe title, servings, and ingredients with quantities and units
- Identify cooking tools used in the video
- Extract step-by-step cooking instructions with timestamps
- Recognize cooking actions and techniques
- Generate structured JSON output of the complete recipe

## System Requirements

- Python 3.8 or higher
- FFmpeg (for video and audio processing)
- CUDA-compatible GPU recommended for faster processing (but not required)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/cookify.git
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
