import argparse
import pandas as pd
import time
from transformers import pipeline
from tqdm import tqdm
import psutil
import torch

# Parse command line arguments
parser = argparse.ArgumentParser(description='Sentiment Analysis with Transformers')
parser.add_argument('--source', type=str, default='messages-binary-classifier.csv', help='Source CSV file containing messages')
parser.add_argument('--score-threshold', type=float, default=0.85, help='Score threshold for classification')
args = parser.parse_args()

# Initialize the sentiment analysis classifier
classifier = pipeline("sentiment-analysis")

# Initialize confusion matrix and timing list
confusion_matrix = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
inference_times = []

# Read the CSV file
df = pd.read_csv(args.source)

# Iterate over the messages in the CSV file with a progress bar
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Analyzing messages"):
    message = row['message']
    expected = row['is_respectful']  # Assuming the column is named 'expected_result'
    
    start_time = time.time()
    result = classifier(message)[0]
    inference_times.append(time.time() - start_time)
    
    # Convert 'True' to 'NEGATIVE' and 'False' to 'POSITIVE' for comparison
    expected_label = 'NEGATIVE' if expected else 'POSITIVE'
    
    # Update confusion matrix based on the prediction and actual result
    if result['label'] == expected_label and result['score'] > args.score_threshold:
        if expected_label == 'NEGATIVE':
            confusion_matrix['TP'] += 1
        else:
            confusion_matrix['TN'] += 1
    else:
        if expected_label == 'NEGATIVE':
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
