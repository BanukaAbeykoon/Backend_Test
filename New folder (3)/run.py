import csv
from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
from flask_pymongo import PyMongo
from waitress import serve
import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt 
from surprise import Dataset,Reader
from surprise import SVDpp
import seaborn as sns
from surprise import Dataset,Reader
from surprise import SVDpp
from sklearn.preprocessing import StandardScaler
# to suppress warnings 
from warnings import filterwarnings
filterwarnings('ignore')


app = Flask(__name__)


app.config["MONGO_URI"] = "mongodb+srv://hiruvidu586:Hirushan2588071M@cluster0.4oftrlo.mongodb.net/test_db?retryWrites=true&w=majority"

csv_file_path = "C:/Users/BANU/Downloads/google_review_ratings.csv"

# mongodb database
mongodb_client = PyMongo(app)
db = mongodb_client.db


@app.route('/lastrecode', methods=['GET'])
def handle_api():
        last_record = db.placesflask.find_one(sort=[('_id', -1)])

        if last_record:
            # Convert the record's _id to a string
            last_record['_id'] = str(last_record['_id'])
            
            # Return the last record as a JSON response
            return jsonify({'placesflask': last_record})

        # Return an appropriate response if there are no records
        return jsonify({'message': 'No records found'})




@app.route('/code', methods=['POST'])
def handle_code():
    
        data = request.get_json()
        user_Id = data.get('User')
        col_1 = data.get('Category 1')
        col_2 = data.get('Category 2')
        col_3 = data.get('Category 3')
        col_4 = data.get('Category 4')
        col_5 = data.get('Category 5')
        col_6 = data.get('Category 6')
        col_7 = data.get('Category 7')
        col_8 = data.get('Category 8')
        col_9 = data.get('Category 9')
        col_10 = data.get('Category 10')
        col_11 = data.get('Category 11')
        col_12 = data.get('Category 12')
        col_13 = data.get('Category 13')
        col_14 = data.get('Category 14')
        col_15 = data.get('Category 15')
        col_16 = data.get('Category 16')
        col_17 = data.get('Category 17')
        col_18 = data.get('Category 18')
        col_19 = data.get('Category 19')
        col_20 = data.get('Category 20')
        col_21 = data.get('Category 21')
        col_22 = data.get('Category 22')
        col_23 = data.get('Category 23')
        col_24 = data.get('Category 24')

        new_user_data = {
        'User': user_Id,
        'Category 1': col_1,
        'Category 2': col_2,
        'Category 3': col_3,
        'Category 4': col_4,
        'Category 5': col_5,
        'Category 6': col_6,
        'Category 7': col_7,
        'Category 8': col_8,
        'Category 9': col_9,
        'Category 10': col_10,
        'Category 11': col_11,
        'Category 12': col_12,
        'Category 13': col_13,
        'Category 14': col_14,
        'Category 15': col_15,
        'Category 16': col_16,
        'Category 17': col_17,
        'Category 18': col_18,
        'Category 19': col_19,
        'Category 20': col_20,
        'Category 21': col_21,
        'Category 22': col_22,
        'Category 23': col_23,
        'Category 24': col_24
    }

        
        result = main(user_Id)
        

        csv_file_path = "C:/Users/BANU/Downloads/google_review_ratings.csv"
        input_data = pd.read_csv(csv_file_path)
        # Load the existing CSV file into a DataFrame
        updated_data = input_data._append(new_user_data, ignore_index=True)

        # Save the updated DataFrame back to the CSV file
        updated_data.to_csv(csv_file_path, index=False)
        return 'Success'

def main(user_Id, col_1, col_2, col_3, col_4, col_5, col_6, col_7, col_8, col_9, col_10, col_11, col_12, col_13, col_14, col_15, col_16, col_17, col_18, col_19, col_20, col_21, col_22, col_23, col_24):
    # Sample data for the new user
    new_user_data = {
        'User': user_Id,
        'Category 1': col_1,
        'Category 2': col_2,
        'Category 3': col_3,
        'Category 4': col_4,
        'Category 5': col_5,
        'Category 6': col_6,
        'Category 7': col_7,
        'Category 8': col_8,
        'Category 9': col_9,
        'Category 10': col_10,
        'Category 11': col_11,
        'Category 12': col_12,
        'Category 13': col_13,
        'Category 14': col_14,
        'Category 15': col_15,
        'Category 16': col_16,
        'Category 17': col_17,
        'Category 18': col_18,
        'Category 19': col_19,
        'Category 20': col_20,
        'Category 21': col_21,
        'Category 22': col_22,
        'Category 23': col_23,
        'Category 24': col_24
    }

    print(new_user_data)

    

    return new_user_data

@app.route('/model', methods=['POST'])
def handle_model():
       data = request.get_json()
       user_Id = data.get('User')
       
       result = main(user_Id)

       placenew = {
        'user_Id' : user_Id,
        'result': result
    }
       db.placesflask.insert_one(placenew)

       return 'Success'

def main(user_Id):
      
    print (user_Id)


    csv_file_path = "C:/Users/BANU/Downloads/google_review_ratings.csv"
    df_review = pd.read_csv(csv_file_path)

    
    column_names = ['User', 'churches', 'resorts', 'beaches', 'parks', 'theatres', 'museums', 'malls', 'zoo', 'restaurants', 'pubs_bars', 'local_services', 'burger_pizza_shops', 'hotels_other_lodgings', 'juice_bars', 'art_galleries', 'dance_clubs', 'swimming_pools', 'gyms', 'bakeries', 'beauty_spas', 'cafes', 'view_points', 'monuments', 'gardens','Unnamed: 25']
    df_review.columns = column_names

    df_review.shape

    df_review.dtypes

    df_review['local_services'].value_counts()

    df_review['local_services'][df_review['local_services'].index == 2712]

    df_review['local_services'][df_review['local_services'] == '2\t2.']

    df_review['local_services'] = df_review['local_services'].replace('2\t2.',2)

    df_review['local_services'] = pd.to_numeric(df_review['local_services'])

    df_review.dtypes

    df_review.describe(include='all')

    df_review[df_review.duplicated()]

    Total = df_review.isnull().sum().sort_values(ascending=False)          

    Percent = (df_review.isnull().sum()*100/df_review.isnull().count()).sort_values(ascending=False)   

    missing_data = pd.concat([Total, Percent], axis = 1, keys = ['Total', 'Percentage of Missing Values'])    
    missing_data

    df_review.drop('Unnamed: 25',axis=1,inplace=True)
    df_review.head()

    df_review['gardens'].mean()

    df_review['gardens'].replace(np.nan,df_review['gardens'].mean(),inplace=True)


    df_review['burger_pizza_shops'].mean()

    df_review['burger_pizza_shops'].replace(np.nan,df_review['burger_pizza_shops'].mean(),inplace=True)


    Total = df_review.isnull().sum().sort_values(ascending=False)          

    Percent = (df_review.isnull().sum()*100/df_review.isnull().count()).sort_values(ascending=False)   

    missing_data = pd.concat([Total, Percent], axis = 1, keys = ['Total', 'Percentage of Missing Values'])    
    missing_data

    df = df_review.copy()

    df_review.drop(columns=['User'], inplace=True)

    fig, ax = plt.subplots(nrows = 8, ncols = 3, figsize=(15, 6))

    plt.tight_layout()

    for variable, subplot in zip(df_review.columns, ax.flatten()):
        
    
        sns.boxplot(df_review[variable], ax = subplot)

    plt.show()

    Q1 = df_review[['churches','resorts','beaches','burger_pizza_shops','hotels_other_lodgings','dance_clubs','swimming_pools','gyms','bakeries','beauty_spas','cafes','view_points','monuments','gardens']].quantile(0.25)

    Q3 = df_review[['churches','resorts','beaches','burger_pizza_shops','hotels_other_lodgings','dance_clubs','swimming_pools','gyms','bakeries','beauty_spas','cafes','view_points','monuments','gardens']].quantile(0.75)

    # Q1, Q3 = Q1.align(Q3, copy=False)
    
    IQR = Q3-Q1

    IQR

    # # df_iqr = df_review.align(Q1, Q3, IQR, axis=1, copy=False) 
    # df_iqr = df_iqr[((df_iqr < (Q1 - 1.5 * IQR)) | (df_iqr > (Q3 + 1.5 * IQR))).any(axis=1)]
    # df_iqr.shape

    df_popularity_table = pd.DataFrame(df_review.mean(),columns=['Average Rating'])
    df_popularity_table['TotalRatingCount'] = df_review.astype(bool).sum(axis=0).values

    bar = df_popularity_table.sort_values(by=['TotalRatingCount'],ascending=True)

    bar = df_popularity_table.sort_values(by=['Average Rating'],ascending=True)

    
    ss = StandardScaler()
    df_scaled = ss.fit_transform(df_review)
    df_scaled = pd.DataFrame(df_scaled,columns=df_review.columns)
    df_scaled.shape

    df_scaled.head()

    

    df_coll_filt_data = df.set_index('User', append=True).stack().reset_index().rename(columns={0:'rating', 'level_2':'Category'}).drop(columns=['level_0'])

    df_coll_filt_data .head(30)

    reader = Reader(rating_scale=(1,5))  # rating scale

    rating_data = Dataset.load_from_df(df_coll_filt_data[['User','Category','rating']],reader)

    trainsetfull = rating_data.build_full_trainset()
    print('Number of user:',trainsetfull.n_users)
    print('Number of items:',trainsetfull.n_items)

    algo = SVDpp(random_state=4)  
    algo.fit(trainsetfull)

    item_id = df_coll_filt_data['Category'].unique()
    item_id

    
    # Get user input for the user ID
    user_id = user_Id

    # Assuming 'item_id' is the list containing item IDs
    item_id = ['churches', 'resorts', 'beaches', 'parks', 'theatres', 'museums',
        'malls', 'zoo', 'restaurants', 'pubs_bars', 'local_services',
        'burger_pizza_shops', 'hotels_other_lodgings', 'juice_bars',
        'art_galleries', 'dance_clubs', 'swimming_pools', 'gyms',
        'bakeries', 'beauty_spas', 'cafes', 'view_points', 'monuments',
        'gardens']  # Replace with your actual list of item IDs

    # Create the test set for the specified user ID
    test_set = [[user_id, iid, 4] for iid in item_id]

    # Display the test set
    print(test_set)


    pred = algo.test(test_set)

    rec = pd.DataFrame(pred).sort_values(by='est', ascending=False)
    print(rec.head(10))

    # Display the 'est' column data
    est_column_data = rec['est']
    print(est_column_data)

    # Calculate the count of 'est' values
    est_count = rec['est'].count()

    # Calculate the average of 'est' values
    est_average = rec['est'].mean()

    # Print the count and average
    print("Count of 'est' values:", est_count)
    print("Average of 'est' values:", est_average)

    # Calculate the average of 'est' values
    est_average = rec['est'].mean()

    # Filter the 'est' values that are above the average
    est_above_average = rec[rec['est'] > est_average]['est']

    # Print the 'est' values above the average
    print("EST values above the average:")
    print(est_above_average,rec)
    

    # Calculate the average of 'est' values
    est_average = rec['est'].mean()

    # Filter the 'est' values and corresponding item names that are above the average
    est_above_average = rec[rec['est'] > est_average][['iid']]



    # Print the 'est' values and item names above the average
    print("EST values and item names above the average:")
    print(est_above_average)
      
    est_average = rec['est'].mean()

    # Filter the 'est' values and corresponding item names that are above the average
    est_above_average = rec[rec['est'] > est_average][['iid']]

    # Create a list of item IDs (iids) from the 'est_above_average' DataFrame
    iids_above_average = est_above_average['iid'].tolist()

    # Print the 'est' values and item names above the average
    print("EST values and item names above the average:")
    print(est_above_average)

    # Print the list of item IDs above the average
    print("Item IDs (iids) above the average:")
    print(iids_above_average)

    return iids_above_average

@app.route('/display', methods=['GET'])
def handle_display():
        last_record = db.placesflask.find_one(sort=[('_id', -1)])

        if last_record:
            # Convert the record's _id to a string
            last_record['_id'] = str(last_record['_id'])
            
            # Return the last record as a JSON response
            return jsonify({'placesflask': last_record})

        # Return an appropriate response if there are no records
        return jsonify({'message': 'No records found'})

@app.route('/csv_record_count', methods=['GET'])
def get_csv_record_count():
    try:
        # Read the CSV file and count the records, including the header row
        with open(csv_file_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            record_count = sum(1 for _ in csvreader)

        return jsonify({"count": record_count})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=5000)


