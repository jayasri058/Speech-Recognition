import os
import torch
import torchaudio
import torch.nn as nn
import torchaudio.transforms as transforms
import matplotlib.pyplot as plt
import streamlit as st

# Load the pre-trained ASR model
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
asr_model = bundle.get_model().eval()

# Placeholder Emotion Classification Model (Replace with actual trained model)
class EmotionClassifier(nn.Module):
    def __init__(self, input_size=128, num_classes=4):
        super(EmotionClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        return self.fc(x)

# Load emotion classification model (Ensure you replace this with a trained model)
emotion_model = EmotionClassifier()
emotion_model.eval()

# Define emotion labels
emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]

# Streamlit UI
st.title("Speech Recognition & Emotion Detection App")
st.write("Upload an audio file (.wav) to transcribe speech and detect emotions.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

if uploaded_file is not None:
    # Save uploaded file
    file_path = "temp_audio.wav"
    #file_path = "E:/InternshipWork/SpeechDetectionSystemsProject/lstm_emotion_model.h5"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    
    # Load and preprocess audio
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
    
    # Display waveform
    fig, ax = plt.subplots()
    ax.plot(waveform.t().numpy())
    ax.set_title("Waveform")
    st.pyplot(fig)
    
    # Display spectrogram
    spectrogram_transform = transforms.MelSpectrogram()
    spectrogram = spectrogram_transform(waveform)
    fig, ax = plt.subplots()
    ax.imshow(spectrogram.log2()[0].numpy(), cmap='viridis', aspect='auto')
    ax.set_title("Spectrogram")
    st.pyplot(fig)
    
    # Run ASR model for transcription
    with torch.inference_mode():
        emissions = asr_model(waveform)
        emissions = torch.log_softmax(emissions[0], dim=-1)
    
    # Decode text
    tokens = torch.argmax(emissions, dim=-1)
    labels = bundle.get_labels()
    predicted_text = "".join([labels[i] for i in tokens[0].tolist() if i < len(labels)])
    
    # Extract features for emotion classification
    with torch.no_grad():
        features = spectrogram.mean(dim=2)  # Simplified feature extraction
        emotion_logits = emotion_model(features.squeeze(0))
        predicted_emotion = emotion_labels[torch.argmax(emotion_logits).item()]
    
    # Display results
    st.audio(file_path, format='audio/wav')
    st.write("### Transcribed Text:")
    st.write(predicted_text)
    st.write("### Predicted Emotion:")
    st.write(predicted_emotion)
    
    # Cleanup
    os.remove(file_path)
