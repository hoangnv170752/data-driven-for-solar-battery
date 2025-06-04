import os
import glob
import json
import psycopg2
from dotenv import load_dotenv

load_dotenv('.env')

FOLDER_DATA = './examples'

conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    port=os.getenv('DB_PORT'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    dbname=os.getenv('DB_NAME')
)
cur = conn.cursor()

for file_path in glob.glob(os.path.join(FOLDER_DATA, '*.json')):
    with open(file_path, 'r') as f:
        data = json.load(f)
        # TODO: Adapt this part to your actual table and data structure
        # Example: insert into batteries table
        # cur.execute(
        #     "INSERT INTO batteries (barcode, protocol, channel_id, cycle_life) VALUES (%s, %s, %s, %s)",
        #     (data['barcode'], data['protocol'], data['channel_id'], data['cycle_life'])
        # )
        print(f"Ingested {file_path}")

conn.commit()
cur.close()
conn.close()
