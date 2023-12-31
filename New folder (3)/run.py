import csv
from flask import Flask, jsonify, request
from flask_pymongo import PyMongo
from waitress import serve
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt 
import seaborn as sns
from surprise import Dataset,Reader
from surprise import SVDpp
from sklearn.preprocessing import StandardScaler
import requests
# to suppress warnings
from warnings import filterwarnings
filterwarnings('ignore')



app = Flask(__name__)


app.config["MONGO_URI"] = "mongodb+srv://hiruvidu586:Hirushan2588071M@cluster0.4oftrlo.mongodb.net/test_db?retryWrites=true&w=majority"

csv_file_path = "/home/ubuntu/Backend_Test/New folder (3)/google_review_ratings.csv"

# mongodb database
mongodb_client = PyMongo(app)
db = mongodb_client.db


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
        

        csv_file_path = "/home/ubuntu/Backend_Test/New folder (3)/google_review_ratings.csv"
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


    csv_file_path = "/home/ubuntu/Backend_Test/New folder (3)/google_review_ratings.csv"
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

    # plt.show()

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

    # df_scaled.head()

    

    df_coll_filt_data = df.set_index('User', append=True).stack().reset_index().rename(columns={0:'rating', 'level_2':'Category'}).drop(columns=['level_0'])

    # df_coll_filt_data .head(30)

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
    # print(test_set)


    pred = algo.test(test_set)

    rec = pd.DataFrame(pred).sort_values(by='est', ascending=False)
    # print(rec.head(10))

    # Display the 'est' column data
    est_column_data = rec['est']
    # print(est_column_data)

    # Calculate the count of 'est' values
    est_count = rec['est'].count()

    # Calculate the average of 'est' values
    est_average = rec['est'].mean()

    # Print the count and average
    # print("Count of 'est' values:", est_count)
    # print("Average of 'est' values:", est_average)

    # Calculate the average of 'est' values
    est_average = rec['est'].mean()

    # Filter the 'est' values that are above the average
    est_above_average = rec[rec['est'] > est_average]['est']

    # Print the 'est' values above the average
    # print("EST values above the average:")
    # print(est_above_average,rec)
    

    # Calculate the average of 'est' values
    est_average = rec['est'].mean()

    # Filter the 'est' values and corresponding item names that are above the average
    est_above_average = rec[rec['est'] > est_average][['iid']]



    # Print the 'est' values and item names above the average
    # print("EST values and item names above the average:")
    # print(est_above_average)
      
    est_average = rec['est'].mean()

    # Filter the 'est' values and corresponding item names that are above the average
    est_above_average = rec[rec['est'] > est_average][['iid']]

    # Create a list of item IDs (iids) from the 'est_above_average' DataFrame
    iids_above_average = est_above_average['iid'].tolist()

    # Print the 'est' values and item names above the average
    # print("EST values and item names above the average:")
    # print(est_above_average)

    # Print the list of item IDs above the average
    # print("Item IDs (iids) above the average:")
    # print(iids_above_average)

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
    
    # Lahiru

@app.route('/apiii', methods=['POST'])
def run_algorithmm():
    # Get request parameters
    data = request.get_json()
    hotelCityy = data.get('hotelCity')
    hotelPricee = data.get('hotelPrice')
   
 
    # Call main function
    result = mainn(hotelCityy, hotelPricee)
 
    hotelnew = {
       
        'result': result
    }
 
    # Save result to MongoDB
 
    db.hotelflask.insert_one(hotelnew)

    return 'Success'

def mainn(hotelCityy,hotelPricee):
    hotelCityy=hotelCityy
    hotelPricee=hotelPricee
    data = pd.read_excel('/home/ubuntu/Backend_Test/New folder (3)/data_hotel.xlsx')
    data['hotel_description'] = data['hotel_description'].fillna('tidak ada deskripsi')
    data[data['price_per_night'].isnull()]
    data.fillna(method='ffill',axis=0,inplace=True)
    review = pd.read_excel('/home/ubuntu/Backend_Test/New folder (3)/review_hotel.xlsx')
    review[review['hotel_id'].isnull()]
    review.dropna(inplace=True)
    review_piv = review.pivot_table(index='hotel_id', aggfunc={'stay_duration':'mean','adults':'mean','children':'mean','rating':['mean','count']})
    review_piv.columns = ['adults','children','rating_count','rating_mean','stay_duration']
    hotel = pd.merge(data,review_piv, on='hotel_id',how='inner')
    c = hotel['rating_mean'].mean()
    hotel['rating_count'].unique()
    m = hotel['rating_count'].quantile(.8)
    weight_rating(hotel,.8)
    hotel['hotel_city'].unique()
    ddd = recomend(hotel,hotelCityy,hotelPricee)
    # a = search_hotel(ddd, m, family='yes',city=hotelCityy,location='all').sort_values(by='price_per_night')
    # print(a)
    hotel_list = ddd[['hotel_name', 'hotel_description', 'hotel_address', 'price_per_night']].to_dict(orient='records')
    return hotel_list
 
 
 
 
def weight_rating(df,var):
    v = df['rating_count']
    R = df['rating_mean']
    C = df['rating_mean'].mean()
    m = df['rating_count'].quantile(var)
    df['score'] = (v/(m+v))*R + (m/(m+v))*C
    return df['score']
 
 
 
def recomend(hotel,hotelCityy,hotelPricee):
    city = hotelCityy.capitalize()
    max_price = hotelPricee
    df = hotel[(hotel['hotel_city']==city) & (hotel['price_per_night'] <= max_price)]
    df = df.sort_values(by='score', ascending = False)
    return df
 
 
 
def search_hotel(hotel, m, family,city,location):
    # chechk for family room
    if family.lower() == 'yes':
        df = hotel[hotel.children != 0]
    else :
        df = hotel
    city
    if city.lower() != 'all':
        df = df[df['hotel_city'].str.lower() == city.lower()]
    else :
        df = df
    # cek location
    if location.lower() == 'all':
        df = df
    else:
        def filter_location(x):
            if location.lower() in str(x).lower():
                return True
            else:
                return False
        df = df.loc[df['hotel_address'].str.split().apply(lambda x: filter_location(x))]
    df = df[df['rating_count'] >= m]
    df = df.sort_values(by=['score'], ascending=False)
    df = df.head()
    return df

@app.route('/displayy', methods=['GET'])
def handle_displayy():
        last_record = db.hotelflask.find_one(sort=[('_id', -1)])

        if last_record:
            # Convert the record's _id to a string
            last_record['_id'] = str(last_record['_id'])
            
            # Return the last record as a JSON response
            return jsonify({'hotelflask': last_record})

        # Return an appropriate response if there are no records
        return jsonify({'message': 'No records found'})


# hirushan

@app.route('/lastrecodee', methods=['GET'])
def handle_api():
        last_record = db.routesflask.find_one(sort=[('_id', -1)])
        if last_record:
            last_record['_id'] = str(last_record['_id'])
            return jsonify({'routesflask': last_record})
        return jsonify({'message': 'No records found'})

@app.route('/lastlocation', methods=['GET'])

def handle_lastlocation():
        last_record = db.locationsflask.find_one(sort=[('_id', -1)])

        if last_record:

            last_record['_id'] = str(last_record['_id'])

            return jsonify({'locationsflask': last_record})
        return jsonify({'message': 'No records found'})


@app.route('/api', methods=['POST'])

def run_algorithm():

    # Get request parameters

    data = request.get_json()

    start_lat = data.get('StartLat')

    start_lon = data.get('StartLon')

    end_lat = data.get('EndLat')

    end_lon = data.get('EndLon')

    temple = data.get('Temples')

    heritages = data.get('Heritages')

    beaches = data.get('Beaches')

    parks = data.get('Parks')

    arts = data.get('Arts')

    username = data.get('Username')

 

    # Call main function

    result = mainnnn(start_lat, start_lon, end_lat, end_lon, temple, heritages, beaches, parks, arts)

 

    routeww = {

        'userName': username,

        'result': result

    }

 

    # Save result to MongoDB

    db.routesflask.insert_one(routeww)

 

    finalresult = result

 

    newresult = get_locations(finalresult)

 

    routewww = {

        'userName': username,

        'places' : newresult

    }

 

    # Save result to MongoDB

    db.locationsflask.insert_one(routewww)

 

    return 'Success'

 

 

 

def mainnnn(StartLat, StartLon, EndLat, EndLon, Temples, Heritages, Beaches, Parks, Arts):

    # print(StartLat)

    # print(StartLon)

    # print(EndLat)

    # print(EndLon)

    # print(Temples)

    # print(Heritages)

    # print(Beaches)

    matrixTemple = np.zeros((1, 2))

    rowsTemple = ['Avarage Distance']

    columnsTemple = ['Japanese Peace Pagoda', 'Shri Sudharmalaya Buddhist Temple\t']

    df1 = pd.DataFrame(matrixTemple, index=rowsTemple, columns=columnsTemple)

   

    matrixHeritageAndHistoricalPlaces = np.zeros((1, 7))

    rowsHeritagesHeritageAndHistoricalPlaces = ['Avarage Distance']

    columnsHeritageHeritageAndHistoricalPlaces = ['Galle Fort', 'Galle Fort Lighthouse\t', 'Dutch Reformed Church', 'Old Dutch Market\t', 'Dutch Hospital Shopping Precinct\t', 'Old Gate', 'Martin Wickramasinghe House & Folk Museum']

    df2 = pd.DataFrame(matrixHeritageAndHistoricalPlaces, index=rowsHeritagesHeritageAndHistoricalPlaces, columns=columnsHeritageHeritageAndHistoricalPlaces)

   

    matrixBeaches = np.zeros((1, 7))

    rowsBeaches = ['Avarage Distance']

    columnsBeaches = ['Jungle Beach', 'Unawatuna Beach', 'Hikkaduwa Beach', 'Dalawella Beach', 'Talpe Beach', 'Ahungalla Beach', 'Induruwa Beach']

    df3 = pd.DataFrame(matrixBeaches, index=rowsBeaches, columns=columnsBeaches)

 

    matrixParks = np.zeros((1, 4))

    rowsParks = ['Avarage Distance']

    columnsParks = ['Beach Park Galle Municipal Council', 'Mahamodara Beach Park Marine Walk', 'The Great wall of lovers galle', 'New Marine Walk Galle Sea sight']

    df4 = pd.DataFrame(matrixParks, index=rowsParks, columns=columnsParks)

   

    matrixArtGallary = np.zeros((1, 2))

    rowsArtGallaries = ['Avarage Distance']

    columnsArtGallaries = ['The Galle Fort Art Gallery', 'Pradeep Pencil Art Gallary']

    df5 = pd.DataFrame(matrixArtGallary, index=rowsArtGallaries, columns=columnsArtGallaries)

    # print("abc")

    data = pd.read_csv("/home/ubuntu/Backend_Test/New folder (3)/DataSet.csv")

   

    # Temples = 3.0

    # Heritages = 4.0

    # Beaches = 6.0

    # StartLat = 6.0265776

    # StartLon = 80.21862858

    # EndLat = 5.9477618

    # EndLon = 80.4519634

    count1 = 0

    count2 = 0

    count3 = 0

    count4 = 0

    count5 = 0

    start_positions = []

    final_start_positions = []

    time = 12

    start_positions.append(StartLat)

    start_positions.append(StartLon)

 

    while time > 0 and (Temples > 0 or Heritages > 0 or Beaches > 0 or Parks > 0 or Arts > 0):

        ProbArr = [Temples, Heritages, Beaches, Parks, Arts]

 

        if getMaxValue(ProbArr) == Temples:

            Temples *= 0.75

 

            if count1 == 0:

                findNeares(matrixTemple, StartLat, StartLon, EndLat, EndLon, df1)

                count1 +=1

               

            column_name_temple = df1.columns[find_min_value_position(matrixTemple)]

           

            galle_data_temple = data[data['Places'] == column_name_temple]

            latitude_col_temple = galle_data_temple['Latitude'].values[0]

            longitude_col_temple = galle_data_temple['Longitude'].values[0]

            start_positions.append(latitude_col_temple)

            start_positions.append(longitude_col_temple)

            # StartLat = latitude_col_temple

            # StartLon = longitude_col_temple

           

            time -= 1

        elif getMaxValue(ProbArr) == Heritages:

            Heritages *= 0.75

           

            if count2 == 0:

                findNeares(matrixHeritageAndHistoricalPlaces, StartLat, StartLon, EndLat, EndLon, df2)

                count2 += 1

           

            column_name_HeritageAndHistoricalPlaces = df2.columns[find_min_value_position(matrixHeritageAndHistoricalPlaces)]

           

            galle_data_column_name_HeritageAndHistoricalPlaces = data[data['Places'] == column_name_HeritageAndHistoricalPlaces]

            latitude_col_HeritageAndHistoricalPlaces = galle_data_column_name_HeritageAndHistoricalPlaces['Latitude'].values[0]

            longitude_col_HeritageAndHistoricalPlaces = galle_data_column_name_HeritageAndHistoricalPlaces['Longitude'].values[0]

            start_positions.append(latitude_col_HeritageAndHistoricalPlaces)

            start_positions.append(longitude_col_HeritageAndHistoricalPlaces)

            # StartLat = latitude_col_HeritageAndHistoricalPlaces

            # StartLon = longitude_col_HeritageAndHistoricalPlaces

 

           

            time -= 1

        elif getMaxValue(ProbArr) == Beaches:

            Beaches *= 0

           

            if count3 == 0:

                findNeares(matrixBeaches, StartLat, StartLon, EndLat, EndLon, df3)

                count3 += 1

           

           

            column_name_Beaches = df3.columns[find_min_value_position(matrixBeaches)]

           

            galle_data_column_name_Beaches = data[data['Places'] == column_name_Beaches]

            latitude_col_Beaches = galle_data_column_name_Beaches['Latitude'].values[0]

            longitude_col_Beaches = galle_data_column_name_Beaches['Longitude'].values[0]

            start_positions.append(latitude_col_Beaches)

            start_positions.append(longitude_col_Beaches)

 

            # StartLat = latitude_col_Beaches

            # StartLon = longitude_col_Beaches

           

           

            time -= 4

        elif getMaxValue(ProbArr) == Parks:

            Parks *= 0

           

            if count4 == 0:

                findNeares(matrixParks, StartLat, StartLon, EndLat, EndLon, df4)

                count4 += 1

           

            column_name_Parks = df4.columns[find_min_value_position(matrixParks)]

           

            galle_data_column_name_Parks = data[data['Places'] == column_name_Parks]

            latitude_col_Parks = galle_data_column_name_Parks['Latitude'].values[0]

            longitude_col_Parks = galle_data_column_name_Parks['Longitude'].values[0]

            start_positions.append(latitude_col_Parks)

            start_positions.append(longitude_col_Parks)

           

            time -= 2

       

 

        else:

            Arts *= 0.75

           

            if count5 == 0:

                findNeares(matrixArtGallary, StartLat, StartLon, EndLat, EndLon, df5)

                count5 += 1

           

            column_name_ArtGallary = df5.columns[find_min_value_position(matrixArtGallary)]

           

            galle_data_column_name_ArtGallary = data[data['Places'] == column_name_ArtGallary]

            latitude_col_ArtGallary = galle_data_column_name_ArtGallary['Latitude'].values[0]

            longitude_col_ArtGallary = galle_data_column_name_ArtGallary['Longitude'].values[0]

            start_positions.append(latitude_col_ArtGallary)

            start_positions.append(longitude_col_ArtGallary)

           

            time -= 2

 

 

 

 

 

 

    # start_positions.append(EndLat)

    # start_positions.append(EndLon)

    #.................................................................................

 

    list_size = len(start_positions)

    arr_size = list_size//2

    # print(arr_size)

    matrixNew = np.zeros((arr_size, arr_size))

    # print(matrixNew)

   

 

    # for i in range(len(start_positions)):

    #     print(start_positions[i], end=" ")

 

    newlen = len(start_positions) // 2

 

    arrz = [[0 for _ in range(newlen)] for _ in range(newlen)]

    for i in range(1):

        for j in range(1 + i, newlen):

           

            url3 = "http://router.project-osrm.org/route/v1/driving/" + str(start_positions[i + i + 1]) + "," + str(start_positions[i + i]) + ";" + str(start_positions[j + j + 1]) + "," + str(start_positions[j + j]) + "?overview=false"

            response3 = requests.get(url3).json()

            distance3 = response3["routes"][0]["distance"]

           

            arrz[i][j] = distance3

           

#             arrz[i][j] = start_positions[i + i] + start_positions[i + i + 1] + start_positions[j + j] + start_positions[j + j + 1]

            #arrz[j][i] = arrz[i][j]

#             print("The shortest road distance between", row_name, "and", column_name, "is", distance1, "meters.")

 

 

    minValue = 0

    # for row in arrz:

    #     for element in row:

    #         print(element, end=" ")

    #     print()  

 

    for j in range(newlen):

        minIndex = j

        for k in range(j + 1, newlen):

            if arrz[0][k] < arrz[0][minIndex]:

                minIndex = k

        # Swap elements in the sorted array

        arrz[0][j], arrz[0][minIndex] = arrz[0][minIndex], arrz[0][j]

        # Swap corresponding j values in the unsorted array

        start_positions[j + j], start_positions[minIndex + minIndex] = start_positions[minIndex + minIndex], start_positions[j + j]

        start_positions[j + j + 1], start_positions[minIndex + minIndex + 1] = start_positions[minIndex + minIndex + 1], start_positions[j + j + 1]

 

    # Print the sorted first row and corresponding j values from the unsorted array

    # print("Sorted first row:")

    for j in range(newlen):

        # print(arrz[0][j], "corresponding j value:", start_positions[j + j])

        # print(arrz[0][j], "corresponding j value:", start_positions[j + j + 1])

        final_start_positions.append(start_positions[j + j])

        final_start_positions.append(start_positions[j + j + 1])

   

    # print(final_start_positions)

 

    #.................................................................................

    final_start_positions.append(EndLat)

    final_start_positions.append(EndLon)

    # print(EndLat)

    # print(EndLon)

    return final_start_positions

 

def getMaxValue(ProbArr):

    max_value = ProbArr[0]

    for value in ProbArr:

        if value > max_value:

            max_value = value

    return max_value

 

def findNeares(disArr, StartLat, StartLon, EndLat, EndLon, df):

    data = pd.read_csv("/home/ubuntu/Backend_Test/New folder (3)/DataSet.csv")

    for i in range(1):

        for j in range(len(disArr[0])):

            column_name = df.columns[j]

            galle_data_column = data[data['Places'] == column_name]

            latitude_col = galle_data_column['Latitude'].values[0]

            longitude_col = galle_data_column['Longitude'].values[0]

           

            row_name = df.index[i]

     

            import requests

            import json

 

            url1 = "http://router.project-osrm.org/route/v1/driving/" + str(longitude_col) + "," + str(latitude_col) + ";" + str(StartLon) + "," + str(StartLat) + "?overview=false"

            response1 = requests.get(url1).json()

            distance1 = response1["routes"][0]["distance"]

             

           

            url2 = "http://router.project-osrm.org/route/v1/driving/" + str(longitude_col) + "," + str(latitude_col) + ";" + str(EndLon) + "," + str(EndLat) + "?overview=false"

            response2 = requests.get(url2).json()

            distance2 = response2["routes"][0]["distance"]

           

            distence = distance1 + distance2

            disArr[i][j] = distance1 + distance2

           

def get_lowest_value(disArr):

    min_value = float('inf')  

 

    for row in disArr:

        for value in row:

            if value < min_value:

                min_value = value

                column_name = df1.columns[1]

 

    return min_value

 

def find_min_value_position(disArr):

    min_value = float('inf')

    min_row = -1

    min_col = -1

 

    for row_idx, row in enumerate(disArr):

        for col_idx, value in enumerate(row):

            if value < min_value:

                min_value = value

                min_row = row_idx

                min_col = col_idx

 

    disArr[0][min_col] = 999999

    return min_col

 

def get_locations(abc):

    data = pd.read_csv("/home/ubuntu/Backend_Test/New folder (3)/DataSet.csv")  

    abc.pop(0)

    abc.pop(0)

   

    new_list_size = len(abc)

    abc.pop(new_list_size-1)

   

    new_list_size = len(abc)

    abc.pop(new_list_size-1)

   

    final_start_places = []

   

    for i in range(0, len(abc), 2):

        latitude = abc[i]

        longitude = abc[i + 1]

 

        # Find matching rows in the DataFrame

        matching_rows = data[(data['Latitude'] == latitude) & (data['Longitude'] == longitude)]

 

        if not matching_rows.empty:

            Places = matching_rows.iloc[0]['Places']

            final_start_places.append(Places)

        else:

            print(f"No matching place found for Latitude: {latitude}, Longitude: {longitude}")

           

    return final_start_places

@app.route('/users', methods=['GET','POST'])

def add_user():

    if request.method == 'GET':

        # Retrieve all users from the database

        users = db.users.find()

        user_list = []

        for user in users:

            user['_id'] = str(user['_id'])

            user_list.append(user)

        return jsonify({'users': user_list})

 

    elif request.method == 'POST':

        data = request.get_json()

        username = data.get('username')

        fristname = data.get('fristname')

        lastname = data.get('lastname')

        gender = data.get('gender')

        age = data.get('age')

        email = data.get('email')

        telephone = data.get('telephone')

        country = data.get('country')

        password = data.get('password')

 

        # Check if name and gender are provided

        if not username or not gender or not age or not email or not telephone or not country or not fristname or not lastname or not password:

            return jsonify({'message': 'All the data must be filled'}), 400

       

        existing_user = db.users.find_one({'username': username})

        if existing_user:

            return jsonify({'message': 'Username already exists'}), 400

 

        # Create a new user document

        user = {

            'username': username,

            'fristname' : fristname,

            'lastname' : lastname,

            'gender': gender,

            'age' : age,

            'email' : email,

            'telephone' : telephone,

            'country' : country,

            'password' : password

        }

 

        # Insert the user into the database

        result = db.users.insert_one(user)

 

        return jsonify({'message': 'User added successfully', 'user_id': str(result.inserted_id)})

 

@app.route('/users/<username>', methods=['GET'])

def get_user(username):

    # Retrieve the user from the database using the given username

    user = db.users.find_one({'username': username})

 

    if not user:

        return jsonify({'message': 'User not found'}), 404

 

    # Convert ObjectId to string for JSON serialization

    user['_id'] = str(user['_id'])

 

    return jsonify({'user': user})

 

 

@app.route('/login', methods=['POST'])

def login():

    data = request.get_json()

    username = data.get('username')

    password = data.get('password')

 

    if not username or not password:

        return jsonify({'message': 'Username and password are required'}), 400

 

    # Retrieve the user from the database based on the given username

    user = db.users.find_one({'username': username})

 

    if not user:

        return jsonify({'message': 'User not found'}), 404

 

    # Check if the password provided matches the password in the database

    if user['password'] != password:

        return jsonify({'message': 'Invalid password'}), 401

 

    # If the username and password are correct, return a success message or any other data you want to provide to the user upon successful login

    return jsonify({'message': 'Login successful', 'user_id': str(user['_id'])})

 

 

@app.route('/users/<username>', methods=['DELETE'])

def delete_user(username):

    # Check if the user with the given username exists in the database

    existing_user = db.users.find_one({'username': username})

 

    if not existing_user:

        return jsonify({'message': 'User not found'}), 404

 

    # Delete the user from the database

    db.users.delete_one({'username': username})

 

    return jsonify({'message': 'User deleted successfully'})

 

@app.route('/users/<username>', methods=['PUT'])

def update_user(username):

    # Retrieve the user from the database based on the given username

    existing_user = db.users.find_one({'username': username})

 

    if not existing_user:

        return jsonify({'message': 'User not found'}), 404

 

    # Get the updated user data from the request

    data = request.get_json()

 

    # Update the user's information

    existing_user['fristname'] = data.get('fristname', existing_user['fristname'])

    existing_user['lastname'] = data.get('lastname', existing_user['lastname'])

    existing_user['gender'] = data.get('gender', existing_user['gender'])

    existing_user['age'] = data.get('age', existing_user['age'])

    existing_user['email'] = data.get('email', existing_user['email'])

    existing_user['telephone'] = data.get('telephone', existing_user['telephone'])

    existing_user['country'] = data.get('country', existing_user['country'])

    existing_user['password'] = data.get('password', existing_user['password'])

 

    # Update the user in the database

    db.users.update_one({'username': username}, {'$set': existing_user})

 

    return jsonify({'message': 'User updated successfully'})

 

if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=5000)


