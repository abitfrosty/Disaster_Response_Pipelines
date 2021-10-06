# Disaster Response Pipeline Project

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
