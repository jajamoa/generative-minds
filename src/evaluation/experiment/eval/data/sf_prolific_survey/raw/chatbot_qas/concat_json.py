import json
import glob

# Define the path pattern for the JSON files
file_pattern = 'original_segs/*.json'

# Initialize an empty list to store all data
all_data = []

# Iterate over each file matching the pattern
for file_name in glob.glob(file_pattern):
    with open(file_name, 'r') as file:
        data = json.load(file)
        all_data.extend(data)  # Concatenate the data

# Write the concatenated data to a new JSON file
with open('concatenated_data.json', 'w') as output_file:
    json.dump(all_data, output_file, indent=2)
