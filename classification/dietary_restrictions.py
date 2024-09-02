import csv
from groq import Groq
from openai import OpenAI
from collections import Counter, defaultdict
import json
import os
from tqdm import tqdm
import time


def get_unique_categories(data_folder, column_name):
    print("Step 1: Extracting unique categories from CSV files...")
    unique_values = set()
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    for filename in tqdm(csv_files, desc="Processing CSV files"):
        with open(os.path.join(data_folder, filename), 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get("approval_status", "").strip().lower() == "approved":
                    value = row.get(column_name, '').strip()
                    if value:
                        unique_values.add(value)

    print("Step 2: Generating standardized categories using Groq...")
    client = Groq()

    prompt = f"""Given the following list of dietary restrictions and food allergies, create a standardized list of category names: {
        ', '.join(unique_values)}. Include 'No restrictions' as a category. Provide the result as a JSON array of strings.
        For example, your response should look something like this: {{"categories": ["one", "two"]}}
        """

    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You're an expert at identifying and categorizing dietary restrictions.
                Your response must be in JSON format."""
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=1024,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )

    categories = json.loads(completion.choices[0].message.content)
    return categories.get('categories', [])


def classify_dietary_restrictions(text, categories, use_openai=False):
    print("Classifying dietary restriction for text:", text)
    client = OpenAI() if use_openai else Groq()

    system_prompt = f"""You're an expert at classifying the description of dietary restrictions using the following categories: {', '.join(
        categories)}.
        If there's no restriction or the text indicates no restrictions, classify it as 'No restrictions'.
        Your respond must be in JSON format where the key is "dietary_restrictions" and the values is a list of categories you classfied from the given user message.
        Your response should look something like this example: {{"dietary_restrictions": ["one", "two"]}}
        There could potentially be more than one category.
        Do not preface anything, just return the JSON object and nothing else.
        """

    user_message = {
        "role": "user",
        "content": text
    }

    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile" if not use_openai else "gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            user_message
        ],
        temperature=0,
        max_tokens=1024,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )

    classification_json = json.loads(
        completion.choices[0].message.content).get("dietary_restrictions", [])
    return classification_json


def get_unique_filename(base_name, extension):
    counter = 1
    while True:
        filename = f"{base_name}({counter}){extension}"
        if not os.path.exists(filename):
            return filename
        counter += 1


def process_csv_files(data_folder, column_name, output_file_base, test_run=False):
    # Get standardized categories
    categories = get_unique_categories(data_folder, column_name)
    print(f"Standardized categories: {categories}")

    classifications = defaultdict(list)

    # Process all CSV files in the data folder
    print("Step 4: Processing CSV files for classification...")
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    for filename in tqdm(csv_files, desc="Classifying entries"):
        with open(os.path.join(data_folder, filename), 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                if test_run and i >= 10:
                    break
                if row.get("approval_status", "").strip().lower() == "approved":
                    text = row.get(column_name, '').strip()
                    if text:
                        try:
                            # Check if text is a list
                            if text.startswith('[') and text.endswith(']'):
                                text_list = json.loads(text)
                                classification = []
                                for item in text_list:
                                    classification.extend(
                                        classify_dietary_restrictions(item, categories))
                                    # Spacer to avoid rapid queries
                                    time.sleep(0.1)
                            else:
                                classification = classify_dietary_restrictions(
                                    text, categories)
                                # Spacer to avoid rapid queries
                                time.sleep(0.1)
                        except Exception as e:
                            print(f"Error processing row in file {
                                  filename}: {text}")
                            print(f"Error: {e}")
                            try:
                                classification = classify_dietary_restrictions(
                                    text, categories, use_openai=True)
                            except Exception as e:
                                print(f"Retry with OpenAI failed: {e}")
                                classification = ['Error']
                    else:
                        classification = ['No restrictions']
                    for cat in classification:
                        classifications[filename].append(cat)

    # Count occurrences of each classification per file
    print("Step 5: Counting occurrences of each classification per file...")
    classification_counts = {filename: Counter(
        cats) for filename, cats in classifications.items()}

    # Get a unique filename for the output
    output_file = get_unique_filename(output_file_base, '.csv')

    # Write results to output CSV
    print("Step 6: Writing results to output CSV...")
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'category_name', 'count'])
        for filename, counts in classification_counts.items():
            for category, count in counts.items():
                writer.writerow([filename, category, count])

    print(f"Results have been saved to {output_file}")


# Example usage
data_folder = './classification/data'
column_name = 'Do you have any dietary restrictions?'
output_file_base = './classification/classification_results'

# Set test_run to True to classify only the first 10 rows
process_csv_files(data_folder, column_name, output_file_base, test_run=False)
