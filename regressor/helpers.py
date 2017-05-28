import numpy as np
import csv


def sse(pred, correct):
    return np.sum((pred - correct)**2)


def get_documents_and_labels(filename):
    print("Loading dataset...")
    print()
    documents = []
    labels = []
    input_indices = [0, 2, 4]   # created_at, domain, title
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            documents.append([row[i] for i in input_indices])
            labels.append(row[1])

    # [1:] = Ignore header of the csv file
    # d[1] = domain
    # d[2] = title
    text_input = ["{} {}".format(d[1], d[2]) for d in documents][1:]
    labels = labels[1:]
    labels = np.asarray(labels).astype(float)

    print("The first entry is:")
    print(text_input[0])
    print()
    return text_input, labels
