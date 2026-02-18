# Lip Reading – Visual Speech Recognition

A deep learning system that predicts spoken words from a silent video by analyzing lip movements.  
The model processes video frames of a speaker’s mouth and outputs the most probable spoken word.

---

## Problem Statement
People with hearing or speech impairments often rely on visual cues to understand speech.  
This project builds a **visual speech recognition (lip-reading) model** that can recognize words without audio input.

---

## Input and Output
**Input:** Video of a person speaking (mouth region frames)  
**Output:** Predicted spoken word/text

Example:
Input: silent video of a person saying "HELLO"  
Output: HELLO

---

## Methodology / Pipeline
1. Extract frames from input video
2. Detect and crop mouth region
3. Resize and normalize frames
4. Pass frame sequence into CNN model
5. Model predicts the spoken word

---

## Model
- Convolutional Neural Network (**CNN**)
- Implemented using **TensorFlow / Keras**
- Trained on **LRS2 (Lip Reading Sentences 2) dataset**

---

## Features
- Video frame extraction using OpenCV
- Image preprocessing and normalization
- Deep learning based word prediction
- Model evaluation and accuracy visualization

---

## Project Structure
```
lip-reading-visual-speech-recognition/
│
├── src/
│   ├── dataset.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── sample_data/
│   └── sample_video.mp4
│
├── results/
│   ├── prediction.png
│   └── training_graph.png
│
├── notebooks/
│   └── experiments.ipynb
│
├── requirements.txt
└── README.md
```

---

## Installation
Clone the repository and install dependencies:

```
pip install -r requirements.txt
```

---

## How to Run
Run prediction on a sample video:

```
python src/predict.py sample_data/sample_video.mp4
```

Expected output:
```
Predicted word: HELLO
```

---

## Dataset
The model is trained on the **LRS2 dataset**, a publicly available dataset containing videos of people speaking in natural conditions.

(Note: Full dataset is not included in this repository due to size limitations.)

---

## Results
- The model is able to recognize common spoken words from silent video
- Training performance and prediction output are shown in the `results/` folder

---

## Applications
- Assistive technology for hearing-impaired individuals
- Silent speech interfaces
- Human-computer interaction
- Speech recognition in noisy environments

---

## Future Improvements
- Use sequence models (LSTM/Transformer)
- Sentence-level prediction instead of word-level
- Real-time webcam prediction

---

## Author
**Tejas Mestry**  
B.E. Computer Science (Data Science)
