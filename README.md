# Voice recognition app

<p align="center">
  <img src="https://github.com/ozxmn/previews/blob/main/reco.gif" alt="Demo" width="100%">
</p>
A desktop application useful for teachers and examiners to recognize the voices of students during both the lessons and exams. Useful to analyze student answers on the lessons and prevent cheating on exams.
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
## Setup & running

1. Download
```text
git clone https://github.com/ozxmn/voice_reco.git

cd voice_recognition
```

2. Install dependencies
```text
pip install -r requirements.txt
```

3. Run the app
```text
python main.py
```
---
## How it works
- Record samples for each person in the dataset (`data/person_name_folder`).
- Train model → creates `model.pkl`.
- Test recognition → identifies who is speaking based on trained data.
---

## Dependencies
- customtkinter
- sounddevice
- soundfile
- numpy
- librosa
- scikit-learn
---

## Gallery
**Start**

![](https://github.com/ozxmn/previews/blob/main/IMG_3742.jpeg)


**Recording**

![](https://github.com/ozxmn/previews/blob/main/IMG_3745.jpeg)

![](https://github.com/ozxmn/previews/blob/main/IMG_3748.jpeg)

![](https://github.com/ozxmn/previews/blob/main/IMG_3749.jpeg)


**Training**

![](https://github.com/ozxmn/previews/blob/main/IMG_3750.jpeg)


**Voice recognition**

![](https://github.com/ozxmn/previews/blob/main/IMG_3751.jpeg)
