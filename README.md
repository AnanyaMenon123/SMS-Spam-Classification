# SMS-Spam-Classification


The "SMS Spam Classification" project is aimed at developing a machine learning model that can accurately classify text messages (SMS) as either spam or non-spam (ham). The primary objective of the project is to help users filter out unwanted or potentially harmful messages, making it easier to manage their communication.
Can also be used for e-mail spam classification.

**Project Overview:**

The project involves several steps, including data preprocessing, feature extraction, model selection, and evaluation. Here's a high-level overview of the project:

1. **Data Collection:**
   Collect a dataset of SMS messages labeled as spam or ham. This dataset serves as the foundation for training and evaluating the machine learning models.

2. **Data Preprocessing:**
   Clean and preprocess the raw text data by removing unnecessary characters, converting text to lowercase, and tokenizing the text into individual words. Stopword removal and stemming is also applied to reduce noise.

3. **Feature Extraction:**
   Convert the preprocessed text data into numerical format that machine learning algorithms can understand. In this project, we have used techniques like Bag of Words (BoW) and TF-IDF (Term Frequency-Inverse Document Frequency) to represent text as feature vectors.

4. **Model Selection:**
   Choose a set of machine learning algorithms that are suitable for text classification tasks. Common algorithms that we included were Naive Bayes, Support Vector Machines, Decision Trees, Random Forests, XGB Boost,KNN and more. 

5. **Model Training and Evaluation:**
   Train each selected model using the preprocessed and feature-extracted data. Evaluate the models' performance using metrics such as accuracy, precision, recall, F1-score, and confusion matrices. 

6. **Model Stacking:**
   Implemented model stacking, which combines the predictions of multiple base models (e.g., Naive Bayes, Support Vector Machines) using a final meta-model (e.g., Random Forest). Stacking can improve classification accuracy by leveraging the strengths of different models.

7. **Deployment:**
   Deployed the trained model to a user-friendly interface using Streamlit. Users can input a text message, and the deployed model will predict whether the message is spam or ham. This makes the project accessible and practical for real-world use.

8. **User Interface (UI):**
   The Streamlit UI provides a simple text input field where users can enter an SMS message. After clicking the "Predict" button, the model's prediction (spam or ham) is displayed on the screen.

**Benefits:**

- Helps users identify and filter out spam messages from their SMS inbox.
- Offers a practical and accessible solution for individuals concerned about spam.


**Project Impact:**

The "SMS Spam Classification" project showcases the power of machine learning in addressing everyday challenges. By accurately identifying spam messages, the project empowers users to manage their SMS communication more effectively, saving time and ensuring a safer messaging experience. 
