import sys
import pandas as pd
import re
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''Load messages and categories from csv tables.
    
    Args:
        messages_filepath (str): Path to messages csv table.
        categories_filepath (str): Path to categories csv table.
        
    Returns:
        df (pandas.DataFrame): Merged dataframe.
    '''
    df_messages = pd.read_csv(messages_filepath, dtype={'id':'int16'}) 
    df_categories = pd.read_csv(categories_filepath, dtype={'id':'int16'})
    df = df_messages.merge(df_categories, on='id')
    return df


def clean_data(df):
    '''Clean and expand df labels of categories into rows.
    
    Args:
        df (pandas.DataFrame): Dataframe with messages and categories.
        
    Returns:
        df (pandas.DataFrame): Cleaned dataframe.
    '''
    df = df.drop_duplicates()
    df_categories = df['categories'].str.split(pat=';', expand=True)
    labels = df_categories.loc[0, :].apply(lambda x: x[:-2]).to_list()
    df_categories.columns = labels
    for column in df_categories.columns:
        df_categories[column] = df_categories[column].apply(lambda x: 0 if x[-1] == '0' else 1)
    df = pd.concat([df, df_categories], axis=1)
    df = df.drop(columns=['categories'])
    # filling duplicated rows that have different labels with maximum value
    df = df.groupby('id').max()
    df = df.reset_index()
    df = df.drop_duplicates()
    # dropping rows that contain nulls
    df = df.dropna(how='all', subset=labels)
    # converting to small int to optimize size
    df['id'] = df['id'].astype('int16')
    df[labels] = df[labels].astype('int8')
    return df


def save_data(df, database_filename):
    '''Insert clean data into SQLite DB.
    
    Args:
        df (pandas.DataFrame): Dataframe with messages and categories.
        database_filename (str): Path to SQLite DB.
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False)


def main():
    '''1. Load data from csv tables.
       2. Clean data.
       3. Save data to SQL.
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
