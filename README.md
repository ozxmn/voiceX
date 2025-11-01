# Voice recognition app

Python voice recognition system using ML.  

---

## Features
- Record and label voice samples  
- Train and save a recognition model (`model.pkl`)  
- Test recognition with confidence scores  
- Clean CustomTkinter GUI  
- Modular structure for easy maintenance

---

## Project structure
```text
voice_recognition/
├── main.py              # Entry point
├── gui.py               # Application GUI
├── audio/
│   ├── recorder.py      # Audio recording
│   └── features.py      # Feature extraction
├── model/
│   ├── trainer.py       # Model training logic
│   └── recognizer.py    # Recognition logic
├── utils/
│   └── types.py         # Dataclasses
├── data/                # Saved WAV files
├── model.pkl            # Saved model (after training)
├── requirements.txt     # List of required Python packages
├── .gitignore           # Git ignore rules
└── README.md            # Project documentation
```
---
## Setup
(*type all of this in the VS Code terminal*)

1️⃣ Download
```text
git clone https://github.com/ozxmn/voice_recognition.git

cd voice_recognition
```

2️⃣ Install dependencies
```text
pip install -r requirements.txt
```

3️⃣ Run the app
```text
python main.py
```
---
## How it works
- Record samples for each person in the dataset (`data/person_name_folder`).
- Train model → creates `model.pkl`.
- Test recognition → identifies who is speaking based on trained data.

⚠️ **Note:** The **Test Recognition** button is disabled until a trained model exists.


## Dependencies
- customtkinter
- sounddevice
- soundfile
- numpy
- librosa
- scikit-learn
---
