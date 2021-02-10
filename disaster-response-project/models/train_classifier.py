import sys
import pandas as pd
from sqlalchemy import create_engine

# download necessary NLTK data
import re
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

import pickle

def load_data(database_filepath):
    """
    Loading data to modeling
    
    Input:
        database_filepath: String, path to database
    Output:
        X: Dataframe, messages from tweets
        Y: Dataframe, tags tweet's classification
        target_names: list, columns tags
    """
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    database = pd.read_sql("SELECT * FROM responses", engine)
    
    #Splitting Dataframe in X and Y
    X = database['message']
    Y = database.drop(columns=['id', 'message', 'original', 'genre'])
    
    #Categories Names
    target_names = Y.columns.tolist()
    
    return X, Y, target_names

def tokenize(text):
    """
    Tokenizing string into list of words and applying lemmatization
    
    Input:
        text: String, string to be treated and tokenized
    Output:
        text: list, list of words tokenized and lemmatized
    """
    
    #Removing Special Characters
    rm_special_char = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    #Tokenize text
    tokens = word_tokenize(rm_special_char)
    
    #Starting Lemmatizator
    lemmatizer = WordNetLemmatizer()
    
    #Applying Lemmatization in Tokens
    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
    
    #Removing stopwords
    clean_tokens = [token for token in clean_tokens if token not in stopwords.words("english")]
    
    return clean_tokens

def build_model(model_type):
    """
    Creates ML pipeline
    
    Input:
        model_type: select an input model type: RandomForestClassifier(), MultinomialNB(), SGDClassifier()
    Returns:
        pipeline: Machine Learning pipeline with fit/predict methods
    """
    
    if model_type == "RandomForestClassifier()":
        print("\tModel RandomForestClassifier Chosen")
        clf = MultiOutputClassifier(RandomForestClassifier())
    elif model_type == "MultinomialNB()":
        print("\tModel MultinomialNB Chosen")
        clf = MultiOutputClassifier(MultinomialNB())
    else:
        print("\tModel RandomForestClassifier Chosen")
        clf = MultiOutputClassifier(RandomForestClassifier())
    
    pipeline = Pipeline(
        [
            ('vect', CountVectorizer(tokenizer=tokenize)), 
            ('tfidf', TfidfTransformer()),
            ('clf', clf)
        ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluating model with classification_report. 
    Metrics: precision, recall, f1-score
    
    Input:
        model: Model, model to be evaluate
        X_test: dataframe, input values to apply in the model and evaluate the results
        Y_test: dataframe, real tags for evaluation of the model
        category_names: list, category names to be evaluated
    Output:
        None
    """
    Y_pred = model.predict(X_test)
    
    for i, target in enumerate(category_names):
        print('{} category metrics: '.format(target))
        print(classification_report(Y_test.iloc[:,i], Y_pred[:,i]))

    pass


def save_model(model, model_filepath):
    """
    Saving the model in pickle file
    
    Input:
        model: Model, model to be saved
        model_filepath: String, filepath to be saved
    Output:
        None
    """
    with open(f'{model_filepath}', 'wb') as f:
        pickle.dump(model, open(f'{model_filepath}', 'wb'))
        print(f"\tSaved {model_filepath} model")
    f.close()
    pass

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(model_type="RandomForestClassifier()")
        
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