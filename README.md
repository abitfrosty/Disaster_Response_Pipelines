# Disaster Response Pipeline Project

![Trained classifier result from input text "Need tents, water and food. Children are sick with fever."](pics/drp_intro.png "Classifier result")

## The goal of this project:

- To train a machine learning model that will classify any input message into 36 categories.

	- The categories are: related, request, offer, aid_related, medical_help, medical_products, search_and_rescue, security, military, child_alone, water, food, shelter, clothing, money, missing_people, refugees, death, other_aid, infrastructure_related, transport, buildings, electricity, tools, hospitals, shops, aid_centers, other_infrastructure, weather_related, floods, storm, fire, earthquake, cold, other_weather, direct_report.
	
## The dataset:

- Data contains 26248 samples and is stored in 2 csv tables: disaster_categories.csv, disaster_messages.csv. The data is imbalanced in such way that some categories are present much frequently ("related" has 20094 occurrences without duplicates) than others ("child_alone" has 0 occurences).

	- Table "disaster_categories.csv" contains columns:
	
		- "id" - Unique identifier of a sample message.
		- "categories" - Labeles (categories) of a sample message.
		
	- Table "disaster_messages.csv" contains columns:
	
		- "id" - Unique identifier of a sample message.
		- "message" - Sample text (input) for a machine learning model.
		- "original" - Sample text before translation into English.
		- "genre" - One of 3 sources the data was collected from.
			- News
			- Direct
			- Social

## Steps to complete:

- ETL

	- Extract from csv tables into Pandas DataFrame.
	- Transform into the multilabeled dataset and clean (remove duplicates). Then dump to SQL (sqlite3) DB.
	- Load from SQL DB.
	
- Machine Learning

	- Pipeline
	
		- Vectorization (creating sparse matrix of vocabulary tokens). Using custom tokenizer:
		
			1. Converting text to lowercase.
			2. Replacing any URLs with "urlplaceholder".
			3. Removing any non-alphanumeric symbols.
			4. Keeping only words that are not in NLTK English stopwords.
			5. Lemmatization
			6. Stemming
			
		- Normalization (Term frequency inverse document frequency) of the values in the vocab tokens matrix on the scale from 0 to 1.
		
		- Classification (applying final estimator as MultiOutputClassifier).
		
	- Training includes choosing the best hyperparameters combination with GridSearchCV.
	
	- Evaluation of the best model on the test set and printing output of classification_report with "precision", "recall" and "f1-score" on each of 36 categories and total summary.
	
- Deployment

	- Website with Flask web framework
	- SQLite database
	- Trained model saved in pickle file
	- Plotly charts
	

## Trained model evaluation:

| Label | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| related | 0.83 | 0.89 | 0.86 | 1002 |
| request | 0.69 | 0.50 | 0.58 | 224 |
| offer | 1.00 | 0.00 | 0.00 | 6 |
| aid_related | 0.71 | 0.63 | 0.67 | 536 |
| medical_help | 0.66 | 0.33 | 0.44 | 94 |
| medical_products | 0.62 | 0.25 | 0.36 | 64 |
| search_and_rescue | 0.64 | 0.16 | 0.26 | 43 |
| security | 1.00 | 0.00 | 0.00 | 29 |
| military | 0.60 | 0.33 | 0.43 | 45 |
| child_alone | 1.00 | 1.00 | 1.00 | 0 |
| water | 0.81 | 0.57 | 0.67 | 87 |
| food | 0.80 | 0.82 | 0.81 | 131 |
| shelter | 0.80 | 0.59 | 0.68 | 119 |
| clothing | 0.71 | 0.80 | 0.75 | 15 |
| money | 0.83 | 0.22 | 0.34 | 23 |
| missing_people | 0.57 | 0.20 | 0.30 | 20 |
| refugees | 0.57 | 0.26 | 0.36 | 50 |
| death | 0.71 | 0.58 | 0.64 | 62 |
| other_aid | 0.73 | 0.14 | 0.24 | 157 |
| infrastructure_related | 1.00 | 0.00 | 0.00 | 75 |
| transport | 0.82 | 0.22 | 0.34 | 65 |
| buildings | 0.52 | 0.39 | 0.45 | 66 |
| electricity | 0.56 | 0.17 | 0.26 | 29 |
| tools | 1.00 | 0.00 | 0.00 | 10 |
| hospitals | 1.00 | 0.00 | 0.00 | 11 |
| shops | 1.00 | 0.00 | 0.00 | 4 |
| aid_centers | 1.00 | 0.00 | 0.00 | 15 |
| other_infrastructure | 1.00 | 0.00 | 0.00 | 54 |
| weather_related | 0.83 | 0.73 | 0.77 | 364 |
| floods | 0.92 | 0.56 | 0.69 | 99 |
| storm | 0.73 | 0.52 | 0.61 | 130 |
| fire | 1.00 | 0.14 | 0.25 | 14 |
| earthquake | 0.89 | 0.82 | 0.85 | 119 |
| cold | 0.73 | 0.42 | 0.53 | 19 |
| other_weather | 0.50 | 0.13 | 0.21 | 68 |
| direct_report | 0.67 | 0.48 | 0.56 | 261 |
|---|---|---|---|---|
| micro avg | 0.77 | 0.58 | 0.67 | 4110 |
| macro avg | 0.79 | 0.36 | 0.41 | 4110 |
| weighted avg | 0.77 | 0.58 | 0.63 | 4110 |
| samples avg | 0.76 | 0.73 | 0.60 | 4110 |
	
### Requirements

- click==8.0.1
- Flask==2.0.1
- greenlet==1.1.1
- itsdangerous==2.0.1
- Jinja2==3.0.1
- joblib==1.0.1
- MarkupSafe==2.0.1
- nltk==3.6.2
- numpy==1.21.2
- pandas==1.3.3
- plotly==5.3.1
- python-dateutil==2.8.2
- pytz==2021.1
- regex==2021.8.28
- scikit-learn==0.24.2
- scipy==1.7.1
- six==1.16.0
- sklearn==0.0
- SQLAlchemy==1.4.23
- tenacity==8.0.1
- threadpoolctl==2.2.0
- tqdm==4.62.2
- Werkzeug==2.0.1

### Instructions

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
		`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
		
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.

    `python run.py`

3. Go to http://0.0.0.0:3001/
