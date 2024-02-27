import torch
print(torch.__version__)

import argparse
import pandas as pd
import time
from transformers import pipeline
from tqdm import tqdm
import psutil

MODEL_PATH = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Parse command line arguments
parser = argparse.ArgumentParser(description='Sentiment Analysis with Transformers')
parser.add_argument('--source', type=str, default='messages-binary-classifier.csv', help='Source CSV file containing messages')
parser.add_argument('--score-threshold', type=float, default=0.85, help='Score threshold for classification')
parser.add_argument('--limit', type=int, default=None, help='Limit the number of predictions to process')
parser.add_argument('--model', type=str, default=MODEL_PATH, help='model to do sentiment analysis with')

args = parser.parse_args()

# Initialize the sentiment analysis classifier
classifier = pipeline("sentiment-analysis", model=args.model, tokenizer=args.model)
test_result = classifier("Hey idiot, can't you figure out this simple problem on your own?")
print(f"Model test: {test_result}")
if args.limit is not None:
    print(f"Limiting to {args.limit} predictions")



# Initialize confusion matrix and timing list
confusion_matrix = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
inference_times = []

# map the model result to the expected label from the data source
result_mappings = {
    'positive': False,
    'neutral': False,
    'negative': True
}

# Read the CSV file
df = pd.read_csv(args.source)
processed_predictions = 0

# Iterate over the messages in the CSV file with a progress bar
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Analyzing messages"):
    processed_predictions += 1
    if args.limit is not None and processed_predictions >= args.limit:
        break

    message = row['message']
    actual_sentiment = row['is_disrespectful']  
    
    start_time = time.time()
    result = classifier(message)[0]
    inference_times.append(time.time() - start_time)
    
    model_result = result_mappings.get(result['label'].lower(), None)
    # Ensure the model result is valid (exists in our mapping)
    if model_result is None:
        print(f"Unexpected model result: {result['label']}, skipping")
        continue  # or handle unexpected result labels as needed
    
    # Determine if the prediction matches the expected label
    correct_prediction = (model_result == actual_sentiment)
    
    # Update confusion matrix based on the prediction and actual result
    if correct_prediction and result['score'] > args.score_threshold:
        if actual_sentiment:
            confusion_matrix['TP'] += 1
        else:
            confusion_matrix['TN'] += 1
    else:
        if actual_sentiment:
            confusion_matrix['FN'] += 1
        else:
            confusion_matrix['FP'] += 1

# Calculate and print the average inference time
average_inference_time = sum(inference_times) / len(inference_times)
print(f"Average inference time per classifier() call: {average_inference_time:.4f} seconds")

# Print out the confusion matrix
print("Confusion Matrix:")
print(f"True Positives: {confusion_matrix['TP']}")
print(f"False Positives: {confusion_matrix['FP']}")
print(f"True Negatives: {confusion_matrix['TN']}")
print(f"False Negatives: {confusion_matrix['FN']}")

# Environment configuration
cpu_info = f"CPU: {psutil.cpu_count(logical=False)} cores, {psutil.virtual_memory().total / (1024 ** 3):.2f} GB RAM"
gpu_info = "GPU: Not available"
if torch.cuda.is_available():
    gpu_info = f"GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"

print(cpu_info)
print(gpu_info)
