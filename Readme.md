# Fake Image Detection using Machine Learning
This project focuses on the development and evaluation of effective deep learning models for detecting fake images. Two models were proposed and evaluated: a customized CNN architecture and a modified VGG16 architecture. 

## Models
1. Customized CNN
2. Extended VGG16

## Dataset

1. deepfake-and-real-images(179.4k images): https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images
2. real-and-fake-face-detection(2041 images): https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection

## Performance Results
**Accuracy:**
1. Customized CNN: 93.15%
2. Extended VGG16: 98.16%

## Installation
1. Clone the repository:
```
git clone https://github.com/your-username/fake-image-detection.git
cd fake-image-detection
```
2. Install the required packages:
```
pip install -r requirements.txt
```
3. Run the Streamlit app:
```
streamlit run app.py
```