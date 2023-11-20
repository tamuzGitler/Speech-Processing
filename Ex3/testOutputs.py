import numpy as np
import pandas as pd

output_filename = "output.txt"

test_labels = pd.read_csv('test_labels')
test_files = test_labels['filename'].to_numpy()
test_true_labels = test_labels['label'].to_numpy().astype(int)

output_dtw = {}
output_euc = {}

with open(output_filename, "r") as file:
    for line in file:
        parts = line.strip().split(" - ")
        if len(parts) == 3:
            audio_file = parts[0]
            output_euc[audio_file] = int(parts[1])
            output_dtw[audio_file] = int(parts[2])
        else:
            print("Invalid line format:", line.strip())

dtw_true = 0
euc_true = 0

for i, test_file in enumerate(test_files):
  true_label = test_true_labels[i]
  if output_dtw[test_file] == true_label:
    dtw_true += 1
  if output_euc[test_file] == true_label:
    euc_true += 1

denominator = len(test_files)
print(f"Euclidian accuracy: {euc_true/denominator}")
print(f"DTW accuracy: {dtw_true/denominator}")