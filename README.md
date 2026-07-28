# 🌱 Plant Disease Recognition System

An AI-powered web application for identifying plant diseases from leaf images using deep learning.

The project enables users to upload an image of a plant leaf and receive an instant prediction along with information about the detected disease. It is designed to assist students, researchers, farmers, and developers interested in computer vision for agriculture.

---

## ✨ Features

- 🔍 Plant disease prediction from leaf images
- 🌿 Supports **38 disease classes**
- 📖 Disease information and recommendations
- ⚡ Fast Streamlit-based web interface
- 📊 Evaluation utilities for model benchmarking
- 📁 Clean and modular project structure
- 🔄 Easily extensible for future models

---

## 📂 Project Structure

```text
Plant-Disease-Recognition/
│
├── main.py                     # Streamlit application
├── disease_info.py             # Disease descriptions
├── requirements.txt
├── environment.yml
│
├── evaluation/
│   └── evaluate_metrics.py
│
├── models/
│   ├── class_names.json
│   ├── metrics.json
│   ├── comparison.csv
│   └── training_config.json
│
├── plant_disease.ipynb         # Training notebook
├── plant_disease_test.ipynb    # Testing notebook
│
└── test/                       # Sample images
```

---

## 🧠 Model

The model is trained on the **PlantVillage** dataset containing approximately **87,000 RGB images** across **38 plant disease categories**.

Dataset split:

| Dataset | Images |
|---------:|-------:|
| Training | 70,295 |
| Validation | 17,572 |
| Test | 33 |

The test folder included in this repository contains sample images that can be used to verify predictions without requiring users to collect their own images.

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yuganshgupta/plant-disease-recognition.git
cd plant-disease-recognition
```

Install dependencies

```bash
pip install -r requirements.txt
```

or create the Conda environment

```bash
conda env create -f environment.yml
conda activate plant-disease
```

---

## ▶️ Running the Application

Start the Streamlit application

```bash
streamlit run main.py
```

Open your browser and navigate to

```
http://localhost:8501
```

---

## 📊 Evaluation

The repository includes an evaluation framework for comparing model performance and analyzing metrics.

Evaluation resources include:

- Performance metrics
- Model comparison
- Training configuration
- Class mappings

---

## 🌿 Supported Crops

The model recognizes diseases affecting crops including:

- Apple
- Blueberry
- Cherry
- Corn (Maize)
- Grape
- Orange
- Peach
- Pepper
- Potato
- Raspberry
- Soybean
- Squash
- Strawberry
- Tomato

along with healthy plant classes.

---

## 🛣️ Roadmap

Future improvements include:

- PyTorch training pipeline
- Transfer learning models
- Additional CNN architectures
- Improved evaluation metrics
- Performance optimization
- Enhanced explainability (Grad-CAM)
- Mobile deployment

---

## 🤝 Contributing

Contributions, feature requests, and suggestions are welcome.

Feel free to fork the repository and submit a pull request.

---

## 📄 License

This project is licensed under the MIT License.

---

## ⭐ Acknowledgements

- PlantVillage Dataset
- TensorFlow & Keras
- Streamlit
- OpenCV
- NumPy
- Pandas
- Scikit-learn

---

If you find this project useful, consider giving it a ⭐ on GitHub!
