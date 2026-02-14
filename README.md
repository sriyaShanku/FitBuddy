# 🏋️ FitBuddy AI - AI Powered Fitness Application

An intelligent fitness application that provides personalized workout and nutrition recommendations using **Local AI** technology. Simply enter your weight, height, and activity level to receive customized fitness guidance powered by a browser-based neural network.

![AI Powered Fitness APP](https://img.shields.io/badge/AI-Local_TensorFlow.js-orange) ![Privacy](https://img.shields.io/badge/Privacy-100%25-green) ![Cost](https://img.shields.io/badge/Cost-Free-blue)

## ✨ Features

- **Local AI Engine**: Get personalized fitness and nutrition advice using a neural network running directly in your browser (TensorFlow.js)
- **100% Private**: Your health data allows stays on your device; no cloud servers involved
- **Dietary Preferences**: Tailored advice for **Vegetarian**, **Non-Vegetarian**, and **Vegan** diets
- **Smart BMI Calculation**: Automatically calculates your BMI and factors it into recommendations
- **Real-time Training**: Watch the AI model train itself on your specific profile in seconds
- **Beautiful Modern UI**: Stunning gradient animations, glassmorphism effects, and smooth transitions
- **Responsive Design**: Works perfectly on all devices - desktop, tablet, and mobile

## 🚀 Quick Start

### Installation & Setup

1. **Download/Clone the files**:
   - Ensure all files (`index.html`, `style.css`, `app.js`) are in the same directory.

2. **Open the Application**:
   - Double-click `index.html` to open it in your default browser.
   - *Alternative*: Run `python -m http.server 8080` for a local server experience.

3. **Get Recommendations**:
   - Enter your weight (in kg)
   - Enter your height (in cm)
   - Select your activity level
   - **Select your Diet** (Veg / Non-Veg / Vegan)
   - Click "**Get Recommendations**"
   - Wait a few seconds for the **Local AI** to train and generate your plan!

## 🏗️ Architecture & How It Works

### Technology Stack

- **Frontend**: Pure HTML5, CSS3, and JavaScript (ES6+)
- **AI Engine**: [TensorFlow.js](https://www.tensorflow.org/js) (Client-Side Machine Learning)
- **Fonts**: Google Fonts (Poppins)

### Data Flow

```
User Input (Weight, Height, Diet)
          ↓
Local Neural Network Training (Browser)
          ↓
Prediction (Calories, Intensity)
          ↓
Expert System Rules (Dietary Logic)
          ↓
Display Results with Animation
```

## 🎨 Design Features

### Visual Elements

1. **Animated Gradient Background**: 
   - 5-color gradient that shifts continuously (Purple/Pink theme)
   - Creates an engaging, dynamic feel

2. **Glassmorphism Card**:
   - Semi-transparent white background with blur effect
   - Premium, modern aesthetic

3. **Responsive Design**:
   - Mobile-first approach that adjusts to any screen size

## 🔒 Security & Privacy

- **Zero Data Leakage**: Since there is no backend server, your personal health info stays in your browser's memory.
- **Offline Capable**: The app does not require an internet connection once loaded.

---


**Built with ❤️ for FitBuddy AI**
*Empowering fitness with privacy-first AI.*
