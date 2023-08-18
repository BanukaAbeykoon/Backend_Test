
import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt 
import seaborn as sns
from surprise import Dataset,Reader
from surprise import SVDpp
# to suppress warnings 
from warnings import filterwarnings
filterwarnings('ignore')

def main():


    # reader = Reader()
    # csv = pd.read_csv('google_review_ratings.csv')

    # # Loading local dataset
    # df_review = Dataset.load_from_df(csv, reader)

   

    csv_file_path = "C:/Users/BANU/Downloads/google_review_ratings.csv"
    df_review = pd.read_csv(csv_file_path)



# Sample data for the new user
new_user_data = {
    'User': 'User 5459',
    'Category 1': 1.7,
    'Category 2': 0,
    'Category 3': 3.0,
    'Category 4': 3.0,
    'Category 5': 5.0,
    'Category 6': 3.0,
    'Category 7': 5.0,
    'Category 8': 2.5,
    'Category 9': 2.5,
    'Category 10': 2.5,
    'Category 11': 1.7,
    'Category 12': 1.7,
    'Category 13': 1.7,
    'Category 14': 1.7,
    'Category 15': 0.5,
    'Category 16': 0.5,
    'Category 17': 0.0,
    'Category 18': 0.5,
    'Category 19': 0.0,
    'Category 20': 0.0,
    'Category 21': 0.0,
    'Category 22': 0.0,
    'Category 23': 0.0,
    'Category 24': 0.0
}

# Load the existing CSV file into a DataFrame
input_data = pd.read_csv(r'C:\Users\BANU\Downloads\google_review_ratings.csv')

# Append the new user data to the DataFrame
input_data = input_data._append(new_user_data, ignore_index=True)

# Save the updated DataFrame back to the CSV file
input_data.to_csv(r'C:\Users\BANU\Downloads\google_review_ratings.csv', index=False)

