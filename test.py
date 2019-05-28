import json

from src.Classifier import Classifier
from src.DBConnector import Connector

config_json = open('./connection.config')
config = json.load(config_json)
connector = Connector(config)

classifier = Classifier(1, connector, './samples/transport_1.csv')