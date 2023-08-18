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
    hotelNamee = data.get('hotelName')
 
    # Call main function
    result = main(hotelCityy, hotelPricee, hotelNamee)
 
    hotelnew = {
       
        'result': result
    }
 
    # Save result to MongoDB
 
    db.hotelTotalflask.insert_one(hotelnew)

    return 'Success'

def main(hotelCityy,hotelPricee,hotelNamee):
    hotelCityy=hotelCityy
    hotelPricee=hotelPricee
    hotelNamee=hotelNamee
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

    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    import nltk
    nltk.download('stopwords')
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    from nltk.corpus import stopwords
    import re

    hotel['desc_all'] = hotel.apply(join_feature,axis=1)
    b = hotel.head()
    print(b)

    clean_spcl = re.compile('[/(){}\[\]\|@,;] ')
    clean_symbol = re.compile('[^0-9a-z #+_]')
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    stopworda = set(stopwords.words('english')) # set stopwords untuk bahasa inggris

    hotel['desc_clean'] = hotel['desc_all'].apply(lambda x: clean_text(x, clean_spcl, clean_symbol, stopword, stopworda))
    c = hotel.head()
    print(c)

    sastrawi = StopWordRemoverFactory()
    stop = sastrawi.get_stop_words()

 

    hotel['desc_clean'] = hotel['desc_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    d= hotel.head()
    print(d)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    # membuat tfidf vectorizer
    tf = TfidfVectorizer()
    tf_matrix = tf.fit_transform(hotel['desc_clean'])
    # membuat matriks similarity score untuk tfidf
    tf_sim = cosine_similarity(tf_matrix,tf_matrix)
    tf_sim
    print(tf_sim)

    # menjadikan nama hotel sebagai anchor index untuk input rekomendasi
    indices = pd.Series(hotel.index, index=hotel['hotel_name']).drop_duplicates()
    # base dataframe sebagai output rekomendasi
    base = hotel.drop(columns=['desc_clean'])
    e = content_recommender(hotel_indices=indices, indices=indices, tf_sim=tf_sim, base=base, hotelNamee = hotelNamee) 
    hotel_list = e[['hotel_name', 'hotel_description', 'hotel_address', 'price_per_night']].to_dict(orient='records')
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
    # city
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

 

def join_feature(x):
    return ''.join(x['hotel_description']) + ' ' + ''.join(x['hotel_province']) + ' ' + ''.join(x['hotel_city']) + ' ' \
            + ''.join(x['hotel_address'])

 

def clean_text(text, clean_spcl, clean_symbol, stopword, stopworda):
    text = str(text)
    text = text.lower() # lowercase text
    text = clean_spcl.sub(' ', text)
    text = clean_symbol.sub(' ', text)
    text = stopword.remove(text) # hapus stopword b. indonesia
    text = ' '.join(word for word in text.split() if word not in stopworda) # hapus stopword b.inggris
    return text

 

def content_recommender(hotel_indices, indices, tf_sim, base, hotelNamee):
    hotel = hotelNamee
    print(hotel)
    
    idx = indices[hotel]
    
    sim_scores = list(enumerate(tf_sim[idx]))
    
    sim_scores = sorted(sim_scores,key=lambda x:x[1],reverse=True)
    
    sim_scores = sim_scores[1:6]
    
    hotel_indices = [i[0] for i in sim_scores]
    hotels = base.iloc[hotel_indices]
    hotels = hotels.sort_values(by = "price_per_night")
    
    input_hotel_row = base[base['hotel_name'] == hotel]
    hotels = pd.concat([hotels, input_hotel_row])

    
    return hotels

if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=5000)
