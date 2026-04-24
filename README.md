# 🌾 AI Crop Recommender System

An intelligent, data-driven web application that recommends the most suitable crops for cultivation based on soil composition and climatic conditions. This project leverages Machine Learning (SVM) to help farmers and agriculturalists maximize yield and efficiency.

---

## 🚀 Overview

The **AI Crop Recommender** is built using **Python**, **Streamlit**, and **Scikit-Learn**. It analyzes specific environmental and soil parameters to provide accurate crop suggestions. The application features a custom-styled, high-contrast UI that eliminates standard Streamlit chrome for a professional, standalone dashboard experience.

### Key Features:
- **Precision Modeling:** Uses a Support Vector Machine (SVM) classifier for high-accuracy predictions.
- **Interactive UI:** Users can adjust soil (N-P-K, pH) and climate (Temp, Humidity, Rainfall) parameters via sliders.
- **Data-Driven:** Trained on the `Crop_recommendation.csv` dataset containing thousands of soil-climate profiles.
- **Clean Interface:** Custom CSS used to hide sidebars and toolbars, focusing purely on the recommendation tool.

---

## 📊 The Dataset

The system is trained on seven key environmental variables:
1.  **N (Nitrogen):** Ratio of Nitrogen content in soil.
2.  **P (Phosphorous):** Ratio of Phosphorous content in soil.
3.  **K (Potassium):** Ratio of Potassium content in soil.
4.  **Temperature:** Ambient temperature in °C.
5.  **Humidity:** Relative humidity in %.
6.  **pH:** The pH value of the soil (acidity/alkalinity).
7.  **Rainfall:** Annual rainfall in mm.

**Target Labels:** The model classifies inputs into various crops including Rice, Maize, Chickpea, Kidney Beans, Pigeon Peas, Moth Beans, Mung Bean, Blackgram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, and Coffee.

---

## 🛠️ Technical Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/)
- **Data Manipulation:** Pandas & NumPy
- **Styling:** Custom CSS & HTML injection

---

## ⚙️ Installation & Setup

To run this project locally, follow these steps:

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/ai-crop-recommender.git](https://github.com/your-username/ai-crop-recommender.git)
cd ai-crop-recommender
