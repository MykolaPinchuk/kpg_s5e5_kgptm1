The objective of this project is to create a simple POC ML model. It is for Kaggle playground Season5 Episode5 dataset. 

You can use kaggle api to pull data for this competition (competition_name = 'playground-series-s5e5'). kaggle.json file is available in current dir.

The main objective of this project is to create POC fast. Do not worry about getting model with very high performance. I suggest using a small subsample to speed up model training. 

You are not allowed to access Kaggle website. You are not allowed to access any code from other people for this specific competition to prevent cheating.

After getting human approval on implementing high-level plan, go ahead and do it. Do not ask human any questions. The final output of this project are predictions on a test set which human will submit for evaluation to Kaggle.


Here is official description of the competition from Kaggle:

Competition: Predict Calorie Expenditure (Playground Series - Season 5, Episode 5) The goal of this competition is to predict the number of calories burned during a workout. This is a regression problem where you will predict a continuous target value. Dataset Details The dataset for this competition was generated from a deep learning model trained on the "Calories Burnt Prediction" dataset. The feature distributions are similar, but not identical, to the original data. Files: train.csv: The training dataset. The Calories column is the target variable. test.csv: The test dataset. Your task is to predict the Calories for each entry. sample_submission.csv: A sample submission file showing the correct format. Data Fields: id: A unique identifier for each entry. Gender: Gender of the participant. Age: Age of the participant in years. Height: Height of the participant in cm. Weight: Weight of the participant in kg. Duration: Workout duration in minutes. Heart_Rate: Average heart rate in beats per minute during the workout. Body_Temp: Body temperature in Celsius during the workout. Calories: Total calories burned (the target variable). Evaluation Metric Submissions are evaluated on the Root Mean Squared Logarithmic Error (RMSLE). This metric is useful for regression tasks where the target has a wide range of values and penalizes underprediction more heavily than overprediction.
