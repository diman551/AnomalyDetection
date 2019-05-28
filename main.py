import sys
import json
import pandas
import logging

from src.DBConnector import Connector
from anomalyDetection import detect

# загрузка параметров подключения
config_json = open('./connection.config')
config = json.load(config_json)

# тип транспорта
transport_type = None
# имя маршрута
route_name = None

# считываем тип транспорта если есть
if len(sys.argv) > 1:
    transport_type = sys.argv[1]
# считываем имя маршрута если есть
if len(sys.argv) > 2:
    route_name = sys.argv[2]

# создание логгера
logger = logging.getLogger('AnomalyDetection')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
fh = logging.FileHandler('detection.log')
fh.setFormatter(formatter)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(consoleHandler)

# подключение к бд
connector = Connector(config, logger)

# таблица с результатами
df = None
# если не указан тип транспорта обходим все данные
if transport_type is None:
    transport_types = connector.getTransportTypes()
    for transport_type in transport_types:
        route_names = connector.getRouteNames(transport_type)
        for route_name in route_names:
            result_df = detect(transport_type, route_name, connector, logger)
            if df is None:
                df = result_df
            else:
                df = pandas.concat(df, result_df)
else:
    # если не указан маршрут обходим весь тип транспорта
    if route_name is None:
        route_names = connector.getRouteNames(transport_type)
        for route_name in route_names:
            result_df = detect(transport_type, route_name, connector, logger)
            if df is None:
                df = result_df
            else:
                df = pandas.concat(df, result_df)
    else:
        df = detect(transport_type, route_name, connector, logger)

#сохраняем результат в таблицу
connector.saveAnomalys(df)