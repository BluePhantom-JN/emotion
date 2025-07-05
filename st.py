import streamlit as st
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
#import matplotlib.pyplot as plt

# Streamlit Title
st.header("IMAGE CAPTURE EMOTIONS")

# Emotion Classes
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Define CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, output):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=0, stride=1)  # 224x224x1 -> 220x220x8
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=3)               # -> 73x73
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0)# -> 71x71x16
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)               # -> 35x35x16
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(35 * 35 * 16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output)

    def forward(self, X):
        X = self.pool1(self.conv1(X))
        X = self.pool2(self.conv2(X))
        X = self.flatten(X)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        return self.fc3(X)

# Load model
model = SimpleCNN(output=7)
model_path = "emotion_data.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

model1_path = "emotion_data_final.pth"
model1 = SimpleCNN(output=7)
model1.load_state_dict(torch.load(model1_path,map_loacation=torch.device('cpu')))

# Image Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        with st.spinner("Loading and processing the image..."):
            img = Image.open(uploaded_file)
            image = transform(img)
            image = image.unsqueeze(0)  # Add batch dimension

            # Predict
            output = model(image)
            probs = torch.softmax(output, dim=1)
            top_probs, top_indices = torch.topk(probs, k=3)

            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption="Original Image", width=200)
            with col2:
                st.subheader("Predicted Emotion")
                st.write(f"Top Emotion: **{emotions[top_indices[0][0]]}**")

                st.subheader("Top 3 Probabilities")
                for i in range(3):
                    emotion_label = emotions[top_indices[0][i]]
                    probability = top_probs[0][i].item()
                    st.write(f"{emotion_label}: {probability:.4f}")
            # Predict2
            output1 = model1(image)
            probs = torch.softmax(output, dim=1)
            top_probs, top_indices = torch.topk(probs, k=3)

            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption="Original Image", width=200)
            with col2:
                st.subheader("Predicted Emotion")
                st.write(f"Top Emotion: **{emotions[top_indices[0][0]]}**")

                st.subheader("Top 3 Probabilities")
                for i in range(3):
                    emotion_label = emotions[top_indices[0][i]]
                    probability = top_probs[0][i].item()
                    st.write(f"{emotion_label}: {probability:.4f}")

    except Exception as e:
        st.error(f"Error: {e}")
