import sys
import pandas as pd
import pickle
import re
from sqlalchemy import create_engine

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


nltk.download(['punkt', 'wordnet', 'stopwords'])


def load_data(database_filepath):
    '''Load the data from the SQL database.
    
    Args:
        database_filepath (str): Path to the SQL database.
        
    Returns:
        X (pandas.Series): Feature of the dataset.
        Y (pandas.DataFrame): Labels of the dataset.
        labels (list): Labels' names.
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', engine)
    
    features = 'message'
    n_labels = 36
    
    X = df.loc[:, features]
    Y = df.iloc[:, -n_labels:]
    labels = Y.columns.to_list()
    
    return X, Y, labels


def tokenize(text):
    '''Clean, normalize, tokenize input text.
    
    Args:
        text (str): Input text to process.
        
    Returns:
        words (list): Tokens.
    '''
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    url_pattern = re.compile(url_regex)
    result = re.sub(url_pattern, ' ', text.lower())
    
    pattern = re.compile(r'[^A-Za-z]+')
    result = re.sub(pattern, ' ', result)
    
    eng_stopwords = stopwords.words('english')
    tokens = word_tokenize(result)
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    words = []
    for word in tokens:
        if word not in eng_stopwords:
            # Lemmatization
            token = lemmatizer.lemmatize(word)
            # Stemming
            token = stemmer.stem(token)
            words.append(token)
            
    return words


def build_model():
    '''Clean, normalize, tokenize input text.
    
    Args:
        text (str): Input text to process.
        
    Returns:
        words (list): Tokens.
    '''
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'clf__estimator__random_state': [0],
        'clf__estimator__splitter': ['random', 'best'],
        'clf__estimator__min_samples_leaf': [2, 4, 8, 16],
        'clf__estimator__min_samples_split': [2, 4, 8, 16, 32, 64]
        }
    model = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tf-idf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(DecisionTreeClassifier()))
            ])
    model = GridSearchCV(model, param_grid=parameters, cv=5, verbose=3)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''Evaluate model on test set and print out metrics of each label.
    
    Args:
        model (GridSearchCV): GridSearchCV instance.
        X_test (pandas.Series): Samples of the test data set.
        Y_test (pandas.DataFrame): Labels of the test data set.
        category_names (list): List with labels names.
    '''
    Y_hat = model.predict(X_test)
    report = classification_report(Y_test.loc[:,:], Y_hat[:,:], zero_division=1, target_names=category_names)
    print(report)


def save_model(model, model_filepath):
    '''Save the trained model to pickle file.
    
    Args:
        model (sklearn.model_selection.GridSearchCV): GridSearchCV instance with the best estimator and parameters.
        model_filepath (str): Destination.
    '''
    pickle.dump(model, open(model_filepath, 'wb'))
    return True


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
            'as the first argument and the filepath of the pickle file to '\
            'save the model to as the second argument. \n\nExample: python '\
            'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

