# 🧠 Smart Cushion AI Engine

### Model Pipeline, Data Mining & Deep Learning Architecture

This repository contains the data science pipeline, machine learning algorithms, Jupyter Notebooks, and datasets required to train the posture classification models utilized by the Fog Node.

<p align="center">
  <b>Convolutional Neural Networks ｜ Sensor Data Normalization ｜ Posture Classification ｜ Scikit-Learn</b>
</p>

---

## 🔗 Project Links

| Item                    | Link                                                                                          |
| ----------------------- | --------------------------------------------------------------------------------------------- |
| 🌐 Project Website      | [https://tonguyentanphuong.github.io/smart-cushion-web/](https://github.com/tonguyentanphuong/smart-cushion-web) |
| 🧑‍💻 AI Repository      | [https://github.com/MuoiVung/smart-cushion-AI](https://github.com/MuoiVung/smart-cushion-AI)      |
| ⚙️ Fog Node Runtime     | [https://github.com/MuoiVung/smart-cushion-fog-node](https://github.com/MuoiVung/smart-cushion-fog-node) |
| ⚙️ Main Architecture    | [https://github.com/MuoiVung/smart-cushion](https://github.com/MuoiVung/smart-cushion)            |

---

## 📌 Project Overview

The core intelligence of CapyCushion lies here. 

Raw data from the 9 FSR sensors is highly non-linear and varies based on the user's weight and the cushion's material. This repository explores various machine learning models—ranging from Random Forests to Convolutional Neural Networks (CNNs)—to accurately classify the 3x3 sensor matrix into 9 distinct seating postures (e.g., Natural Upright, Lean Forward, Contralateral Rotation).

This repository handles the **Offline Training Phase**. Once a model achieves high validation accuracy, the `.h5` weights and `.pkl` scalers are exported and deployed to the Fog Node for real-time inference.

This project includes:
* 📊 **Data Mining** — Outlier detection, correlation heatmaps, and dataset balancing.
* 🧠 **Model Training** — Building Keras CNNs and Sklearn FNNs/Random Forests.
* 📈 **Evaluation** — Confusion matrices and accuracy reports.
* 📦 **Exporting** — Generation of production-ready model artifacts.

---

## 🛠️ Technology Stack

| Layer              | Tools / Components                             |
| ------------------ | ---------------------------------------------- |
| Language           | Python 3.9+                                    |
| Environment        | Jupyter Notebook                               |
| Deep Learning      | TensorFlow, Keras                              |
| Machine Learning   | Scikit-Learn, XGBoost                          |
| Data Processing    | Pandas, NumPy                                  |
| Visualization      | Matplotlib, Seaborn                            |

---

## 💡 Motivation

Posture detection using simple thresholding on pressure sensors is notoriously inaccurate due to the massive variability in human body types.

Using an AI approach allows the system to:
* 🪑 Generalize across different seat materials (hard wood vs. soft cushions).
* 🧍 Generalize across users of different weights.
* 📐 Detect complex combinations of leaning and twisting that simple logic rules would miss.

We chose a **CNN** over an FNN because the 3x3 FSR matrix effectively acts as a 9-pixel grayscale image, meaning spatial relationships between adjacent sensors are highly indicative of posture.

---

## 🧩 AI Pipeline & Architecture

### Model Workflow

| Phase       | Action                                                       |
| ----------- | --------------------------------------------------------------- |
| **Data Mining**| Load raw human-trial CSV files (`lab_14052026/`). Remove outliers and visualize correlation heatmaps. |
| **Scaling**  | Apply Scikit-Learn `MinMaxScaler(feature_range=(0, 1))`. Export the scaler as `.pkl`. |
| **Structuring**| Reshape the 9 flat sensor features into a 3x3 grid, treating it as a grayscale image. |
| **Training** | Feed data through Keras Conv2D, MaxPooling, and Dense layers (`CNN.ipynb`). |
| **Evaluation**| Generate confusion matrices and calculate validation accuracy across 9 states. |
| **Export**   | Save the finalized weights as a `.h5` file for deployment to the Fog Node. |

---

## 🗂️ Repository Structure

```text
smart-cushion-AI/
│
├── README.md
├── CNN.ipynb                  (Primary CNN Training Pipeline)
├── Data_Mining + FNN.ipynb    (Alternative Feedforward NN Pipeline)
├── Data_Mining + CNN.ipynb    (Exploratory Data Analysis & CNN)
│
├── train_rf.py                (Random Forest Baseline Script)
├── train_xgb.py               (XGBoost Baseline Script)
│
├── *.h5                       (Exported Keras Model Weights)
├── *.pkl                      (Exported Scikit-Learn Scalers)
│
├── ai/                        (Raw training dataset CSVs)
└── lab_14052026/              (Lab collected validation dataset)
```

---

## 🚀 How to Train & Deploy a New Model

### 1. Prerequisites
Ensure you have Python installed and run:
```bash
pip install tensorflow scikit-learn pandas numpy jupyter matplotlib seaborn
```

### 2. Running the Pipeline
1. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open `CNN.ipynb`.
3. Run the cells sequentially: Load Data → Normalize → Train → Evaluate.

### 3. Exporting & Deployment
1. At the end of the notebook, the code automatically saves the `.h5` model and `.pkl` scaler.
2. **Critical Step:** Copy *both* the `.h5` and `.pkl` files.
3. Paste them into the `models/` directory of your `smart-cushion-fog-node` repository.
4. Restart the Fog Node or use the Fog GUI to select the new model for hot-reloading.


---

## 🎯 Conclusion
By treating the 3x3 FSR matrix as an image and utilizing a Convolutional Neural Network, the CapyCushion AI Engine achieves over 96% accuracy across 9 distinct posture classes. The separation of this offline training environment from the Fog runtime allows for rapid prototyping, robust data mining, and seamless hot-reloading of new AI models.
