import json
import csv
from tqdm import tqdm

# File paths
json_file_path = 'data/amazon/books.json'
csv_file_path = 'data/amazon/books.csv'

# Count the number of lines in the JSON file (optional for better accuracy in the progress bar)
total_lines = sum(1 for _ in open(json_file_path, 'r'))

# Open the JSON file
with open(json_file_path, 'r') as json_file:
    # Open the CSV file for writing
    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['asin', 'user_id', 'rating', 'timestamp'])

        # Process each line in the JSON file
        for line in tqdm(json_file, total=total_lines, desc="Processing JSON", unit="lines"):
            try:
                record = json.loads(line)
                
                # Using the correct key for timestamp
                timestamp = record.get('timestamp', 0)
                if timestamp != 0:
                    asin = record.get('asin', '')
                    user_id = record.get('user_id', '')
                    rating = record.get('rating', 0)
                    writer.writerow([asin, user_id, rating, timestamp])
                    
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")

print(f'Data has been saved to {csv_file_path}')


