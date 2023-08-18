from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
from flask_pymongo import PyMongo
from waitress import serve
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)


app.config["MONGO_URI"] = "mongodb+srv://hiruvidu586:Hirushan2588071M@cluster0.4oftrlo.mongodb.net/test_db?retryWrites=true&w=majority"


# mongodb database
mongodb_client = PyMongo(app)
db = mongodb_client.db


@app.route('/apii', methods=['POST'])
def run_algorithm():
    # Get request parameters
    data = request.get_json()
    hotelCityy = data.get('hotelCity')
    hotelPricee = data.get('hotelPrice')
   
 
    # Call main function
    result = main(hotelCityy, hotelPricee)
 
    hotelnew = {
       
        'result': result
    }
 
    # Save result to MongoDB
 
    db.hotelflask.insert_one(hotelnew)

    return 'Success'

def main(hotelCityy,hotelPricee):
    hotelCityy=hotelCityy
    hotelPricee=hotelPricee
    data = pd.read_excel('C:/Users/BANU/Downloads/data_hotel.xlsx')
    data['hotel_description'] = data['hotel_description'].fillna('tidak ada deskripsi')
    data[data['price_per_night'].isnull()]
    data.fillna(method='ffill',axis=0,inplace=True)
    review = pd.read_excel('C:/Users/BANU/Downloads/review_hotel.xlsx')
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
    recomend(hotel,hotelCityy,hotelPricee)
 
    a = search_hotel(hotel, m, family='yes',city='surabaya',location='all').sort_values(by='price_per_night')
    print(a)
    hotel_list = a[['hotel_name', 'hotel_description', 'hotel_address', 'price_per_night']].to_dict(orient='records')
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
    df = df.sort_values(by='score', ascending = False).head()
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
def handle_display():
        last_record = db.hotelflask.find_one(sort=[('_id', -1)])

        if last_record:
            # Convert the record's _id to a string
            last_record['_id'] = str(last_record['_id'])
            
            # Return the last record as a JSON response
            return jsonify({'hotelflask': last_record})

        # Return an appropriate response if there are no records
        return jsonify({'message': 'No records found'})
 

if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=5000)
