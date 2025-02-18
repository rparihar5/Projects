Black Friday Sales Prediction Project

Problem Statement:
Black Friday Sale is one of the most significant shopping events globally, where customers come out and make purchases at discounted prices. Retailers are eager to understand customer behavior during this period to optimize their sales and maximize profits. Accurate predictions of customers purchase amounts to understand customers behavior can help the retailers in inventory management, target marketing, and personalized customer experiences. This project addresses the problem by building predictive models that estimate the purchase amounts based on historical sales data and customer profiles.

Data Description

The dataset used for this project is taken from Kaggle competition. This is a huge dataset which contains 550,068 instances with features. The data provides comprehensive information about the customers, products, and their purchase behavior during the Black Friday Sales at a retail supermarket.
Feature Information
1.	User_ID: A unique identifier for each customer.
2.	Product_ID: A unique identifier for each product.
3.	Gender: Gender of the customers. 
4.	Age: Age group of customers. 
5.	Occupation: Occupation code of the customer, with each number representing a different occupation type.
6.	City_Category: The category of the city where the customers reside.
7.	Stay_In_Current_City_Years: The number of years the customer has stayed in the current city.
8.	Marital_Status: Marital status of the customer.
9.	Product_Category_1, Product_Category_2, Product_Category_3: These columns represent different product category purchase by the customers.
10.	 Purchase: The purchase amount spend by the customer during the Black Friday Sales. This is also our target variable, which are going to predict.

Methodology

For this project, we implemented a systematic machine learning pipeline to forecast sales prediction figures for Black Friday event. Our main objective was to create, train, and assess a model that aligns with the specific business context.
1)	Identifying the Business Problem:
The first and most important thing in machine learning pipeline is to know what you are trying to solve. Here, we are trying to predict the purchase amount for the customers during the Black Friday Sales at a retail store. Understanding customer spending behavior during such a significant sales period is crucial for retailers aiming to optimize marketing strategies and manage inventory effectively.

2)	Data Selection and Preparation:
•	The dataset used in this project is huge dataset having 550,068 instances and 12 columns. 
•	 Handling Missing Values: We addressed missing values in the “Product_Category_2 and Product_Category_3 columns by applying median imputation. This approach minimized the potential biases that could arise from missing data, thereby preserving the integrity of the dataset.
•	 Encoding Categorical Variables: To facilitate the processing of categorical data such as Gender, Age, and City_Category, we employed one-hot encoding and label encoding techniques. This conversion allowed our machine learning models to interpret these features effectively.
•	Feature Scaling: Numerical features, particularly the target variable Purchase, were scaled to ensure they were on a similar range. This step was essential for models that are sensitive to feature scaling, as it ensured that no single feature dominated the learning process.

3)	Model Development and Selection:

•	Supervised Learning Approach:
As our dataset is labeled which means it a supervised dataset, we opted for supervised learning methodology. Supervised learning is well-suited for predicting outcomes based on known input-output pairs, making it ideal for our regression task.
•	Model Selection:
As this is a regression task, we have tried to solve the problem with different algorithms, and then we have chosen the one which had the best accuracy. We have used:
1.	Linear Regression
2.	Decision Tree Regressor
3.	Random Forest Regressor
4.	XGBoost Regressor

4)	Data Splitting and Validation:
Train-Test Split: The dataset was divided into a training set (80%) and a testing set (20%). This division allowed us to train the model on one portion of the data while evaluating its performance on unseen data, ensuring the model’s ability to generalize.
Cross-Validation: To ensure our model we validated it by applying K-Fold cross-validation. This technique was crucial in assessing the model’s performance across different data subsets, helping to detect overfitting and ensuring that the model’s predictions are robust and reliable.
Feature Engineering: Here we have performed engineering on each feature and tried enhancing the model’s performance. This included creating interaction terms and aggregating metrics.

5)	Model Training Optimization
Training the models: Each model was trained using the training set, with performace evaluation conduced on the testing set for each model to know how all models are performing. We have used several evaluation metrics, including the R ²score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE), for accuracy and reliability for each model.

6)	Evaluation and Model Selection:

Comparing Model Performance: We compared the performance of all the models based on their evaluation metrics. The XGBoost Regressor emerged as the most accurate, achieving an R ² score of 0.71, indicating that it could explain 71% of the variance in the target variable.
7)	Conclusion:
The XGBoost Regressor, with its advanced boosting techniques, provided the most accurate predictions for Black Friday sales. The insights derived from this model can be leveraged by retailers to enhance marketing strategies, personalize customer experiences, and efficiently manage inventory during peak sales periods like Black Friday.


WORKFLOW OF THE MACHINE LEARNING MODEL
We have used Jupyter Notebook, python kernel to develop our model and further work with our dataset.
