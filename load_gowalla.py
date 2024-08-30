import csv
from tqdm import tqdm

# File paths
txt_file_path = 'data/gowalla/Gowalla_totalCheckins.txt'
csv_file_path = 'data/gowalla/gowalla.csv'

# Count the number of lines in the text file (optional for better accuracy in the progress bar)
total_lines = sum(1 for _ in open(txt_file_path, 'r'))

# Open the text file
with open(txt_file_path, 'r') as txt_file:
    # Open the CSV file for writing
    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['user_id', 'item_id', 'rating', 'timestamp'])

        # Process each line in the text file
        for line in tqdm(txt_file, total=total_lines, desc="Processing TXT", unit="lines"):
            try:
                # Split the line into components based on tab separation
                components = line.strip().split('\t')
                
                user_id = components[0]
                timestamp = components[1]
                item_id = components[4]
                rating = 4  # Fixed rating value
                
                # Remove 'T' and 'Z' from the timestamp to make it a proper format
                formatted_timestamp = timestamp.replace('T', ' ').replace('Z', '')

                writer.writerow([user_id, item_id, rating, formatted_timestamp])
                    
            except Exception as e:
                print(f"Error processing line: {e}")

print(f'Data has been saved to {csv_file_path}')


