from flask import Flask, jsonify
from waitress import serve
import csv

app = Flask(__name__)

# CSV file path
csv_file_path = "C:/Users/BANU/Downloads/google_review_ratings.csv"

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
