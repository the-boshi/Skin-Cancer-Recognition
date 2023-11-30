import os
import pandas as pd
import random
import csv

seed = 2
random.seed(seed)

test_examples = train_examples = validation_examples = 0


for line in open("dataset/labels.csv").readlines()[1:]:
    split_line = line.split(",")
    img_file = split_line[0]
    row = [img_file+'.jpg', 
           int(float(split_line[1])), 
           int(float(split_line[2])), 
           int(float(split_line[3])), 
           int(float(split_line[4])), 
           int(float(split_line[5])), 
           int(float(split_line[6])), 
           int(float(split_line[7])), 
           int(float(split_line[8])), 
           int(float(split_line[9]))]
    
    random_num = random.random()

    if random_num < 0.85:
        with open('dataset/train.csv', 'a') as train:
            writer = csv.writer(train)
            writer.writerow(row)
        train_examples += 1
    else:
        with open('dataset/test.csv', 'a') as test:
            writer = csv.writer(test)
            writer.writerow(row)
        test_examples += 1

print(f"Number of training examples {train_examples}")
print(f"Number of test examples {test_examples}")