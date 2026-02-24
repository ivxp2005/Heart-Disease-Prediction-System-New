# ❤️ Heart Disease Risk Prediction System

A machine learning-powered web application that predicts the 10-year risk of coronary heart disease using the Framingham Heart Study dataset. Built with Python, Scikit-learn, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Screenshots](#screenshots)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements a comprehensive heart disease prediction system that:
- Analyzes 15 medical and demographic risk factors
- Predicts 10-year coronary heart disease risk
- Provides an interactive web interface for real-time predictions
- Visualizes risk factors and prediction confidence
- Uses the renowned Framingham Heart Study dataset

**Key Achievement:** Achieved **84.97% accuracy** using Logistic Regression with proper feature engineering and hyperparameter tuning.

---

## ✨ Features

### 🔮 Prediction System
- **Real-time Risk Assessment**: Instant prediction of heart disease risk
- **Confidence Scoring**: Displays prediction probability for transparency
- **Multiple Input Methods**: User-friendly sliders and number inputs
- **Smart Validation**: Conditional logic (e.g., cigarettes/day appears only for smokers)

### 📊 Data Visualization
- **Interactive Risk Gauge**: Visual representation of risk level
- **Feature Breakdown**: Bar chart showing contribution of each risk factor
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Dark Theme UI**: Professional appearance with large, readable fonts

### 🎨 User Experience
- **Clean Interface**: Organized into tabs (Prediction, Model Info, About)
- **Educational Content**: Explains each risk factor and model details
- **Instant Feedback**: Color-coded results (green for low risk, red for high risk)
- **Export Ready**: Results can be saved or shared

---

## 📊 Dataset

**Source:** Framingham Heart Study  
**Size:** 4,238 patients  
**Features:** 15 clinical and demographic attributes  
**Target:** 10-year risk of coronary heart disease (CHD)

### Input Features:
1. **Demographics:**
   - Sex (Male/Female)
   - Age (29-70 years)
   - Education Level (1-4)

2. **Behavioral Factors:**
   - Current Smoker (Yes/No)
   - Cigarettes Per Day (0-70)

3. **Medical History:**
   - Blood Pressure Medication (Yes/No)
   - Prevalent Stroke (Yes/No)
   - Prevalent Hypertensive (Yes/No)
   - Diabetes (Yes/No)

4. **Clinical Measurements:**
   - Total Cholesterol (107-696 mg/dL)
   - Systolic Blood Pressure (83-295 mmHg)
   - Diastolic Blood Pressure (48-142 mmHg)
   - BMI (15.54-56.80)
   - Heart Rate (44-143 bpm)
   - Glucose Level (40-394 mg/dL)

---

## 🏆 Model Performance

### Logistic Regression (Selected Model)
- **Accuracy:** 84.97%
- **Precision:** High precision for minority class
- **Recall:** Balanced performance
- **F1-Score:** Optimized for medical use case
- **Training Time:** Fast and efficient
- **Interpretability:** Excellent (clear feature importance)

### Alternative Models Tested:
| Model | Accuracy | Status | Notes |
|-------|----------|--------|-------|
| **Logistic Regression** | **84.97%** | ✅ **Selected** | Best balance of accuracy and interpretability |
| Random Forest | 82.3% | ❌ Overfitted | High variance on test set |
| XGBoost | 81.8% | ❌ Overfitted | Complex, less interpretable |

**Why Logistic Regression?**
- Superior generalization to unseen data
- Fast inference for real-time predictions
- Clear interpretability for medical stakeholders
- Robust performance with proper regularization

---

## 🛠️ Technologies Used

### Core Technologies:
- **Python 3.8+** - Primary programming language
- **Streamlit** - Web application framework
- **Scikit-learn** - Machine learning library
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing

### Data Visualization:
- **Plotly** - Interactive charts and gauges
- **Matplotlib** - Statistical visualizations
- **Seaborn** - Enhanced data visualization

### Model Deployment:
- **Joblib** - Model serialization
- **Pickle** - Feature and metadata storage

### Development Tools:
- **Jupyter Notebook** - Exploratory data analysis
- **Git** - Version control
- **VS Code** - Development environment

---

## 💻 Installation

### Prerequisites:
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Step-by-Step Instructions:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR-USERNAME/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. **Create a virtual environment:**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import streamlit; print(streamlit.__version__)"
   ```

---

## 🚀 Usage

### Running the Web Application:

1. **Activate virtual environment** (if not already active):
   ```bash
   # Windows
   .venv\Scripts\activate

   # macOS/Linux
   source .venv/bin/activate
   ```

2. **Launch Streamlit app:**
   ```bash
   streamlit run heart_disease_app.py
   ```

3. **Open your browser:**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in terminal

### Using the Application:

1. **Input Patient Data:**
   - Fill in all 15 risk factors using the sidebar form
   - Use sliders for continuous variables
   - Select Yes/No for categorical variables

2. **Get Prediction:**
   - Click the "🩺 Predict Risk" button
   - View risk level (High/Low) with confidence score
   - Examine the risk gauge and feature breakdown

3. **Explore Model Information:**
   - Switch to "Model Information" tab
   - Learn about the algorithm and evaluation metrics
   - Understand how predictions are made

4. **Read About the Project:**
   - Navigate to "About" tab
   - View dataset description and feature explanations

---

## 📁 Project Structure

```
heart-disease-prediction/
│
├── heart_disease_app.py          # Main Streamlit web application
├── heart_disease_model.pkl       # Trained Logistic Regression model
├── model_info.pkl                # Model metadata and training info
├── feature_names.pkl             # List of feature names for prediction
├── framingham.csv                # Framingham Heart Study dataset
│
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── README.md                     # This file
│
└── .venv/                        # Virtual environment (not in repo)
```

### Key Files:
- **`heart_disease_app.py`** - Complete Streamlit application with UI and prediction logic
- **`heart_disease_model.pkl`** - Serialized Logistic Regression model (trained)
- **`model_info.pkl`** - Contains accuracy, feature importance, and training details
- **`feature_names.pkl`** - Ordered list of features for consistent predictions
- **`framingham.csv`** - Original dataset used for training

---

## 📸 Screenshots

### Main Prediction Interface
The clean, dark-themed interface provides an intuitive user experience with:
- Large header and clear instructions
- Organized sidebar for input parameters
- Real-time prediction with confidence score
- Interactive risk gauge visualization

### Risk Assessment Results
- **Low Risk:** Green gauge, reassuring message, probability < 50%
- **High Risk:** Red gauge, cautionary message, probability ≥ 50%
- **Feature Breakdown:** Bar chart showing which factors contribute most to risk

---

## 🔮 Future Improvements

### Short-term Enhancements:
- [ ] Add PDF report export functionality
- [ ] Implement batch prediction for multiple patients
- [ ] Include feature importance explanations for each prediction
- [ ] Add data validation with medical range checks

### Medium-term Features:
- [ ] Deploy to cloud platform (Streamlit Cloud, Heroku, or AWS)
- [ ] Create REST API for integration with electronic health records
- [ ] Add user authentication and patient history tracking
- [ ] Implement A/B testing for model improvements

### Long-term Vision:
- [ ] Train on larger, more diverse datasets
- [ ] Incorporate deep learning models (Neural Networks)
- [ ] Add time-series analysis for progression tracking
- [ ] Multi-language support for global accessibility
- [ ] Mobile app version (iOS/Android)

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes:**
   ```bash
   git commit -m "Add some AmazingFeature"
   ```
4. **Push to the branch:**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Contribution Guidelines:
- Write clear, commented code
- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation as needed
- Test thoroughly before submitting

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 👤 Author

**Your Name**
- GitHub: [@YOUR-USERNAME](https://github.com/YOUR-USERNAME)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/your-profile)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- **Framingham Heart Study** - For providing the comprehensive cardiovascular dataset
- **Streamlit Team** - For the excellent web framework
- **Scikit-learn Contributors** - For robust machine learning tools
- **Open Source Community** - For inspiration and support

---

## 📞 Support

If you have any questions or run into issues:
1. Check the [Issues](https://github.com/YOUR-USERNAME/heart-disease-prediction/issues) page
2. Create a new issue with detailed description
3. Contact via email (see Author section)

---

## ⭐ Star This Repository

If you find this project useful, please consider giving it a star! It helps others discover the project.

---

**Made with ❤️ and Python**
