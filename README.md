# Eldercare Agent

An AI-powered conversational companion system designed for elderly care, featuring multimodal emotion recognition and personalized response generation.

## Features

- Text and voice-based conversation interface
- Real-time emotion analysis using text and audio inputs
- Dynamic response generation with empathy and personalization
- Early warning system for emotional health monitoring
- User profile management and conversation history tracking
- Wellbeing questionnaire integration
- Emotion trend visualization

## Technology Stack

- Backend: Flask web framework
- Emotion Recognition: BERT-based text analysis + voice feature extraction
- Machine Learning: Stacking ensemble models (MLP, LightGBM, Ridge regression)
- Database: MongoDB with file storage fallback
- Frontend: HTML/CSS/JavaScript with real-time audio recording

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open your browser and navigate to `http://localhost:5000`

## Project Structure

- `app.py` - Main Flask application entry point
- `emotion_detection/` - Emotion recognition models and training scripts
- `strategy/` - Response generation and conversation strategy modules
- `user_bio/` - User profile and conversation management
- `templates/` - HTML templates for web interface
- `static/` - CSS and client-side assets

## Usage

The system provides both text and voice conversation modes. Users can register, log in, and engage in conversations while the system monitors emotional states and provides appropriate responses based on detected emotions and conversation context. 