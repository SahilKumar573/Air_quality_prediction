🎯 Objective
 Predict pollutant concentration (target variable)
 Compare multiple ML algorithms
 Build an interactive UI for prediction
 Help in early air quality monitoring

🧠 Machine Learning Algorithms Used
✅ Linear Regression
✅ Decision Tree Regressor
✅ Random Forest Regressor (Best Model)

📂 Dataset
The dataset contains environmental features such as:
Input Features
Temperature
Humidity
Wind Speed
NO₂
SO₂
CO
Target Variable
Pollutant Concentration (e.g., PM2.5 / AQI)

🛠️ Technologies Used
Python 🐍
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Joblib
Streamlit (for UI)

📁 Project Structure
Air Quality/
│
├── data/
│   ├── city_hour1.csv
│   ├── city_hour2.csv
│   └── ...
│
├── train_model.py      # Model training
├── ui.py               # Streamlit UI
├── air_quality_model.pkl
├── requirements.txt
└── README.md
⚙️ Installation
Step 1: Clone or download project
Step 2: Install dependencies
pip install -r requirements.txt
🚀 How to Run
▶️ Step 1: Train the model
python train_model.py

This will create:

air_quality_model.pkl
▶️ Step 2: Run the UI
streamlit run ui.py

Then open the browser link shown in terminal.

📊 Model Performance

Example metrics (will vary with data):

Linear Regression → Moderate accuracy

Decision Tree → Better than linear

Random Forest → ⭐ Best performance

Evaluation metrics used:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

R² Score

🎨 Features of UI

Easy input sliders

Real-time prediction

Clean dashboard

User-friendly interface

🔮 Future Improvements

Use real-time API data

Deploy on cloud

Add AQI category prediction

Deep Learning models

Time-series forecasting