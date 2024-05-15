# About HandLingo
HandLingo is a pose-based translator for American Sign Language (ASL). Users are able to record video clips of themselves performing single signs, which are detected and subsequently translated into text of their language of choice. Supported languages include Chinese (Simplified), Latin, and German. Due to model constraints, gloss recognition is enabled only for the first ten glosses of the WLASL dataset.

## Getting started

### Prerequisites
This was developed in Python 3.9. To install the necessary packages using pip, please run `pip install -r requirements.txt`
or ensure that all version requirements in `requirements.txt` are met. Your environment should also have access to a webcam and microphone.

### Installation
1. Clone the repo:
```
git clone https://github.com/catlu11/HandLingo.git
```
2. Run `python app.py`.

## File structure
```
.
├──app.py: main entry point of the application and contains UI, video recording, and overall program logic
├──translation.py: class containing translator functions
├──videos: folder containing user-recorded video clips
│  ├── ...
├──gloss_recognition: folder containing files necessary for automated gloss recognition
│  ├──Transformer_nih.py: contains a Transformer-specific model class and functions to interface with app.py
│  ├──wlasl_class_list.txt: list of gloss classes and their indices
│  └──Transformer: folder containing files specific to the Transformer model
│     ├──extract_features.py: functions for extracting pose features from video clips
│     ├──transformer.py: contains classes that define the Transformer model architecture
│     └──best_epoch_wrist.pt: load file for Transformer model parameters
├──requirements.txt: list of Python package requirements
├──README.md
└──LICENSE
```
