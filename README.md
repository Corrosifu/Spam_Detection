# Spam Detection

## Description

The goal of this project is to provide a historical overview of the various methods that have been used over time to handle spam classification in emails or messages.

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Results](#results)
- [Features](#features)
- [Roadmap](#roadmap)
- [Authors](#authors)
- [License](#license)

---

## Installation

1. Clone the repository:

git clone https://github.com/Corrosifu/Spam_Detection.git

2. Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate # on Linux/Mac
venv\Scripts\activate # on Windows

3. Install dependencies:

pip install -r requirements.txt


---

## Usage
If you want to work on CEAS data
- Open and Launch the [**CEAS_Data_clear**](https://github.com/Corrosifu/Spam_Detection/blob/main/CEAS-08.ipynb)  files to analyse and preprocess data.
- Open and Launch the [**Filter_model**](https://github.com/Corrosifu/Spam_Detection/blob/main/CEAS-08/Filter_model_CEAS.ipynb) file to evaluate the classic filter performances.
- Open and Launch the [**ML_model**](https://github.com/Corrosifu/Spam_Detection/blob/main/CEAS-08/ML_model_body.ipynb) and [**ML_model_all_features**](https://github.com/Corrosifu/Spam_Detection/blob/main/CEAS-08/ML_model_multiple_features.ipynb) files to evaluate standard ML methods performances such as Logistic Regression or Random Forest.
- Open and Launch the [**LLM_finetuning**](https://github.com/Corrosifu/Spam_Detection/blob/main/CEAS-08/LLM_finetuning_body.ipynb) and [**LLM_finetuning_all_features**](https://github.com/Corrosifu/Spam_Detection/blob/main/CEAS-08/LLM_finetuning_multiple_features.ipynb) files to evaluate how LLM perform on Spam vs Ham classification Task.
- The [**Functions**](https://github.com/Corrosifu/Spam_Detection/blob/main/Functions.py) file holds all functions designed to simplify the notebooks code.
  
If you want to work on kaggle data it is the same usage

  ## Data
  
  1. Data sources:
    
  [CEAS_08](https://github.com/Corrosifu/Spam_Detection/blob/main/Data/CEAS-08)  [Kaggle](https://www.kaggle.com/datasets/abdmental01/email-spam-dedection) 

  2. Data Analysis (The detail is provided only for Kaggle dataset but will be updated soon):
 
     ### Spam(1) and Ham(0) distribution 
     
     ![3a878c1d-4020-442a-a522-b8f81afcbd12](https://github.com/user-attachments/assets/55d46441-ad9f-4d07-a2de-da0954634be9)

     ### Word Frequency among Spam and Ham
     
    
    ![7e096731-05af-49c5-9e8d-f3862ec46be9](https://github.com/user-attachments/assets/7c0543c8-efc1-456b-8bb5-c6f617971f32)


  ## Results

  ### Filter Methods

  

     
  
  


