A Comprehensive Measure of Well-Being using Machine Learning

Project Overview

This project predicts the Human Development Index (HDI) using Machine Learning techniques. The application analyzes important socio-economic indicators such as Life Expectancy, Mean Years of Schooling, and Gross National Income (GNI) per Capita to estimate the Human Development Index. A Flask-based web application provides a simple user interface where users can enter these indicators and obtain predicted HDI values instantly.

Objectives:
Predict Human Development Index (HDI).
Perform data preprocessing and visualization.
Train a Linear Regression model.
Evaluate model performance.
Deploy the trained model using Flask.
Provide a simple web interface for predictions.

Features:
Dataset preprocessing
Missing value handling
Data visualization
Linear Regression model
Model evaluation using R² Score and Mean Squared Error
Model serialization using Pickle
Flask web application
User-friendly prediction interface

Technology Stack:
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Flask
HTML
CSS
Pickle

Project Workflow:
Environment Setup
Import Libraries
Dataset Collection
Data Exploration
Data Visualization
Data Preprocessing
Feature Selection
Train-Test Split
Linear Regression Model Training
Model Evaluation
Save Model
Flask Deployment

Dataset
The dataset contains Human Development Index indicators collected from international development reports.

Main attributes include:
Country
Life Expectancy at Birth
Mean Years of Schooling
Expected Years of Schooling
Gross National Income Per Capita
Human Development Index
Machine Learning Algorithm

Linear Regression
The model learns the relationship between socio-economic indicators and the Human Development Index to predict HDI for new user inputs.


Installation:
Clone the repository
git clone https://github.com/yourusername/A-Comprehensive-Measure-of-Well-Being.git

Install dependencies
pip install -r requirements.txt

Run the application
      python app.py


Project Structure
dataset/
models/
src/
templates/
static/
app.py
requirements.txt
README.md


Future Enhancements
Support multiple ML algorithms
Improve prediction accuracy
Interactive dashboards
Cloud deployment
User authentication
Real-time analytics


Team Members:
Nagella Sravani
Nagalli Deepak
Nallagoti Lavanya
Ugadi Shiva Kumar
Vannala Madhusekhar

Conclusion:
This project demonstrates the application of Machine Learning in predicting the Human Development Index. By integrating data preprocessing, visualization, model development, and Flask deployment, it provides an end-to-end solution for HDI prediction. The system is scalable and can be extended with advanced machine learning algorithms and cloud deployment for real-world use.

