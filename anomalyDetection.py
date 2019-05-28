from src.Classifier import Classifier
from src.DBConnector import Connector

def detect(transport_type:int, route_name:str, connector:Connector, logger=None):
    # загрузка классификатора
    classifier = Classifier.load(transport_type=transport_type, connector=connector, logger=logger)

    # загрузка информации о маршруте
    route_df = connector.getRouteData(route_name)
    data = route_df[["ON_DAY", "SUM_VALUES",
                    "MAX(AGE1_SEX0)", "MAX(AGE2_SEX0)", "MAX(AGE3_SEX0)",
                    "MAX(AGE1_SEX1)", "MAX(AGE2_SEX1)", "MAX(AGE3_SEX1)", ]].values
    route_nums = route_df[["ROUTE_NUMBER"]].values[:, 0]

    # получение меток классов
    labels = classifier.classify(data)

    # выделение аномальных данных
    anomaly_df = classifier.getAnomalys(data, route_nums, labels)

    if logger is not None: logger.info('Route ' + route_name + ' detected ' + str(len(anomaly_df)) + ' anomalies in data')
    return anomaly_df