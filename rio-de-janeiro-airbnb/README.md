# Disaster Response Pipeline Project

### Project

In this project, it was developed an Exploratory Data Analysis with the Rio de Janeiro AirBNB dataset!

### File Structure

	- datasets
	| - listings.csv
	| - calendar.csv
	| - reviews.csv
  
	- README.md

### Python Libraries

- pandas
- numpy
- matplotlib.pyplot
- re
- seaborn

### Business Understanding

The EDA was develop with a public dataset of the Rio de Janeiro AirBNB informations. It's available in the folowwing link: http://insideairbnb.com/get-the-data.html

It was develop an EDA to answer the following questions: 

1. Is there a price difference in the different types of accommodation?
2. How is the behavior of rental prices in the city of Rio de Janeiro during the year?
3. How is the price behavior if separated by accommodation types?

### Data Understanding

The database has three main tables: 

- **listings.csv**: is the main database in which you have the information of the houses for rent within airbnb (many information are available in this dataset such as price of the day, reviews, cleaning note etc ...).

- **calendar.csv**:  is the database with the house price calendar! Imagine that when we rent a house through AirBnb we can see how much it costs to rent the house on a certain day!

- **reviews.csv**: is the database with the comments of the reviews (In this EDA we will not use this base)

### Prepare Data: 

- It was used just `listings.csv` and `calendar.csv` files in order to develop the EDA project
- To make all the calculations it was necessary to transform the column `price` to Float type because originally it was String Type.
- In `calendar.csv` the column `date` was transformed to datetime to be used by time series plots.
- In `calendar.csv`, it was removed some null values from `price` columns (it was an impact of 0.1% of total base)

### Data Modeling: 

- ML Algorithms weren't developed in this project. 
- It was used data wrangling techniques, plots and decriptive statistics to make the analysis.


### Evaluate the Results: 

1. **Is there a price difference in the different types of accommodation?**

As we can see for Entire home/apt:
- at least 50% of houses rent prices are less than 350,00
- at least 75% of houses rent prices are less than 698,00 
- the mean value of houses rent prices are 823,00! 

As we can see for Hotel room:
- at least 50% of houses rent prices are less than 290,00
- at least 75% of houses rent prices are less than 183,00 
- the mean value of houses rent prices are 325,00! 

As we can see for Private room:
- at least 50% of houses rent prices are less than 290,00
- at least 75% of houses rent prices are less than 160,00 
- the mean value of houses rent prices are 448,00! 

As we can see for Shared Room:
- at least 50% of houses rent prices are less than 200,00
- at least 75% of houses rent prices are less than 100,00 
- the mean value of houses rent prices are 1456,00! (Outlier!) 

If we'll travel to the Rio de Janeiro City we can know better if the price is very high or not to the types of accomodation for rent!

As we can see Private Room and Shared Room are the cheapest prices for rent, the most expensive is Entire home/apt! 

2. **How is the behavior of rental prices in the city of Rio de Janeiro during the year?**

It was notice that the price houses increases after the second semester of the year and has the hightest prices in Christmas and New Year!
There are some spikes in prices during all the year, firstly the hypothesis it was because the prices can change when it's a weekend or weekday, but we found some differents evidences that's not true (in the next question).

3. **How is the price behavior if separated by accommodation types?** 

Different types of accomodations has differents behaviors in prices over the year, generally the Entire Home/Apt dictates the behavior of the prices, when we see the Hotel Room prices we notice that the price increases just after the Christmas and New Year. For the Private Room the prices maintain a stability with a slight decrease in price over the year and at last the Shared Room seems to have some outliers points that generates some spikes in prices.


