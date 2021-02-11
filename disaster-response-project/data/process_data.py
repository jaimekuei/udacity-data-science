import sys
import pandas as pd
import re

from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loading input data to dataframe
    
    Input:
        messages_filepath: String, path to messages csv file
        categories_filepath: String, path to categories csv file
    Output:
        database: Dataframe, merged messages and categories databases
    """
    
    #Loading csv files to dataframe (messages and categories)
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #Merging datasets
    dataset = messages.merge(categories, how='left', on='id')
    
    return dataset

def extract_column_name(text):
        
    """
    Extract just numbers, letters and "_" symbol in a text input
    
    Input:
        text: String, input string to be treated
    Output:
        text: String, output string treated
    """
    
    text = re.sub(r"[^a-zA-Z_]", "", text)
    return text

def clean_data(df):
    
    """
    Cleaning dataframe, transform categories columns into columns targets, remove duplicates and treat "related" column
    
    Input:
        df: Dataframe, input dataframe to be treated
    Output:
        database: Dataframe, cleaned dataframe
    """
    
    
    #### TREATING CATEGORIES COLUMNS ####
    #Treating categories column
    categories = df['categories'].str.split(pat=";", expand=True)
    
    #extracting categories columns names
    categories_values = categories.head(1).values.tolist()[0]
    categories_columns = list(map(extract_column_name, categories_values))
    
    #Treating categories columns names
    categories.columns = categories_columns
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    #dropping categories columns from original dataframe 
    dataset = df.drop(['categories'], axis = 1)
    dataset = dataset.merge(categories, left_index=True, right_index=True)
    
    #removing duplicates
    dataset = dataset.drop_duplicates()
    
    #Treating "related" columns: value 2 to become 0
    dataset.loc[dataset['related']==2, 'related'] = 0
    
    return dataset


def save_data(df, database_filename):
    
    """
    Saving dataframe in SQLAlchemy database
    
    Input:
        df: Dataframe, dataframe to be saved in sql database
        database_filename: String, database file name
    Output:
        None
    """
    
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql("responses", engine, index=False, if_exists = 'replace')
    pass  


def main():
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