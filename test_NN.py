import pandas
from src.Classifier import Classifier

classifier = Classifier(1, None, "./samples/transport_1.csv")

print(classifier.accuracy)
