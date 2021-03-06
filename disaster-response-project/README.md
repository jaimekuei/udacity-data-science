# Disaster Response Pipeline Project

### Project

In this project, it was developed an end-to-end ML pipeline to analyze the Figure Eight's database that has informations about disaster messages.

Some process were developed as: 
1. Data Engineering - extract, transform and load data (figure eight database)
2. ML pipeline - build a model using the disaster database
3. Web Development - front-end that can classify a message

### File Structure

	- app
	| - template
	| |- master.html
	| |- go.html
	|- run.py

	- data
	|- disaster_categories.csv
	|- disaster_messages.csv
	|- process_data.py
	|- InsertDatabaseName.db

	- models
	|- train_classifier.py
	|- classifier.pkl

	- README.md

### Instructions during the project:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
