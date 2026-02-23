<h1>Flower Classification Web Application | Flask, PyTorch, EfficientNet</h1>

- Built a full-stack Flask web application to classify 102 flower species using a fine-tuned EfficientNet-B0 deep learning model

- Implemented Top-5 prediction ranking with confidence scores and real-time image inference

- Designed a responsive UI supporting image upload and preview with integrated prediction results

- Trained and evaluated the model on the Oxford Flowers-102 dataset, applying transfer learning and regularization techniques

- Integrated external knowledge retrieval to display flower descriptions from Wikipedia

- Managed large model artifacts using Git LFS and followed ML deployment best practices

##  Features

- **Accurate Classification** – Fine-tuned EfficientNet-B0 trained on Oxford Flowers-102  
- **Top-5 Predictions** – Displays confidence-ranked flower predictions  
- **Rich Information** – Fetches flower descriptions from Wikipedia (when available)  
- **User-Friendly Interface** – Clean, responsive UI built with Flask, HTML, and CSS  
- **Image Preview** – Uploaded image shown alongside prediction results  
- **Flexible Input** – Supports file upload and base64-encoded image data  

---

##  Model Details

- **Architecture:** EfficientNet-B0 (pre-trained on ImageNet)
- **Custom Layers:** Batch Normalization and Dropout
- **Input Size:** 224 × 224 pixels
- **Output Classes:** 102 flower categories

---




##  Setup & Installation

### Prerequisites
- **Python 3.8+**
- **pip** (Python package manager)
- **CUDA-capable GPU** *(optional, for faster inference)*

---

### Installation 

- **Clone the repository**
  ```bash
  git clone https://github.com/hailhydrov2/Flower-Prediction.git
  cd Flower-Prediction
