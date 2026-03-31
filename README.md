## This is my first Machine Learning project
This being my first project, I made an application that takes as input a few parameters and predicts the math marks that the student scored
# 🌾 Remaining Useful Life Prediction using Weather Data

A Machine Learning project that predicts the **remaining useful life (RUL)** of perishable produce using **temperature and humidity time-series data**.

> Built by Parth Wadhwa (IIT Delhi)

---

## 🌟 Problem Statement

Perishable goods (like fruits and vegetables) degrade over time depending on environmental conditions such as **temperature and humidity**.

Accurately predicting the **remaining shelf life** can help:
- Reduce food waste  
- Improve supply chain decisions  
- Optimize storage conditions  

---

## 🧠 Solution

This project builds a **time-series based ML model** that:

- Takes sequential weather data (temperature & humidity)
- Learns degradation patterns over time
- Predicts the **remaining useful life (in hours)**

---

## ⚙️ Approach

### 🔹 Data Processing
- Time-series sequence creation (sliding window)
- Normalization using scaler
- Feature engineering from environmental data

---

### 🔹 Model Architecture
- Deep learning model (LSTM / GRU)
- Captures temporal dependencies in weather data
- Trained on sequential input (e.g., last 48 hours)

---

### 🔹 Prediction Pipeline
1. Input: Temperature & humidity sequence  
2. Preprocessing using trained scaler  
3. Model inference  
4. Output: Remaining Useful Life (hours)

---

## 📊 Example

```python
predict_rul(model, scaler, temp_seq, rh_seq, sequence_length=48)
