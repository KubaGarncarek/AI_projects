import csv
import sys
import datetime
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}") 
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)
        evidence = []
        labels = []
        data = (evidence, labels)
        for row in reader:
            evidence.append([
                int(row[0]), float(row[1]), int(row[2]), float(row[3]),
                int(row[4]), float(row[5]), float(row[6]), float(row[7]),
                float(row[8]), float(row[9]), get_month_number(row[10]),
                int(row[11]), int(row[12]), int(row[13]), int(row[14]),
                int(row[15] == "Returning_Visitor"), int(row[16] == "TRUE")
            ])
            labels.append(int(row[17] == "TRUE"))
    print(labels)
    return data

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    
    return model.fit(evidence,labels)
    


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    sensitivity_incorect = 0
    sensitivity_corect = 0
    specificity_incorect = 0
    specificity_corect = 0
    print(len(predictions))
    for actual, predicted in zip(labels, predictions):
        if actual == 1:
            
            if int(actual) == int(predicted):
                sensitivity_corect +=1
            else:
                sensitivity_incorect +=1
        else:
            if actual == predicted:
                specificity_corect +=1
            else:
                specificity_incorect +=1
    print(sensitivity_corect)
    sensitivity = round(float((sensitivity_corect)/(sensitivity_corect+sensitivity_incorect)),4)
    specificity = round(float((specificity_corect)/(specificity_corect+specificity_incorect)),4)

    return (sensitivity, specificity)
    

def get_month_number(month):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return months.index(month)


if __name__ == "__main__":
    main()
