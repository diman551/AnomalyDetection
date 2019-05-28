from sqlalchemy import create_engine
import pandas
from pandas import DatetimeIndex
from pandas import Series


class Connector:
    def __init__(self, config, logger=None):
        self.logger = logger

        ip = config["ip"]
        port = config["port"]
        user = config["user"]
        password = config["password"]
        tnsname = config["tnsname"]

        try:
            self.connection = create_engine("oracle+cx_oracle://"+user+":"+password+"@"+ip+":"+port+"/?service_name="+tnsname)
        except Exception as e:
            content = (tnsname + ' is Unreachable,The reason is ' + str(e)).strip()
            print(content)
        else:
            self.cursor = self.connection

    def executeDataFrame(self, sql):
        return pandas.read_sql(sql=sql, con=self.connection)

    def getTransportTypes(self):
        if self.logger is not None: self.logger.info('Transport types is required')
        df = self.executeDataFrame("SELECT UNIQUE TRANSPORT_TYPE FROM FUZZY_SEARCH.PASSENGERS_FLOW_DATA")
        if self.logger is not None: self.logger.info('Transport types is loaded')
        return df.values[:, 0]

    def getRouteNames(self, transport_type):
        if self.logger is not None: self.logger.info('Route names is required')
        df = self.executeDataFrame(
            "SELECT UNIQUE ROUTE_NAME "
            "FROM FUZZY_SEARCH.PASSENGERS_FLOW_DATA "
            "WHERE TRANSPORT_TYPE = " + str(transport_type))
        if self.logger is not None: self.logger.info('Route names is loaded')
        return df.values[:, 0]

    def getRouteData(self, route_name):
        if self.logger is not None: self.logger.info('Route ' + route_name + ' is required')
        sql = """WITH ROUTE_TABLE AS (
                        SELECT * FROM FUZZY_SEARCH.PASSENGERS_FLOW_DATA WHERE ROUTE_NAME = '""" + route_name + """'
                    )
                    SELECT OPERATOR_ID, ROUTE_NUMBER,  ROUTE_NAME, ON_DATE, DEPART_ID, ARRIVE_ID, SUM(VALUE) AS SUM_VALUES,
                           MAX(AGE1_SEX0), MAX(AGE2_SEX0), MAX(AGE3_SEX0), MAX(AGE1_SEX1), MAX(AGE2_SEX1), MAX(AGE3_SEX1)
                    FROM ROUTE_TABLE
                        JOIN (
                            SELECT ROUTE_NUMBER, SUM(VALUE) AS AGE1_SEX0
                            FROM ROUTE_TABLE
                            WHERE AGE_RANGE_ID = 1 AND SEX_ID = 0
                            GROUP BY ROUTE_NUMBER
                        ) P10_DATA USING(ROUTE_NUMBER)
                        JOIN (
                            SELECT ROUTE_NUMBER, SUM(VALUE) AS AGE2_SEX0
                            FROM ROUTE_TABLE
                            WHERE AGE_RANGE_ID = 2 AND SEX_ID = 0
                            GROUP BY ROUTE_NUMBER
                        ) P20_DATA USING(ROUTE_NUMBER)
                        JOIN (
                            SELECT ROUTE_NUMBER, SUM(VALUE) AS AGE3_SEX0
                            FROM ROUTE_TABLE
                            WHERE AGE_RANGE_ID = 3 AND SEX_ID = 0
                            GROUP BY ROUTE_NUMBER
                        ) P30_DATA USING(ROUTE_NUMBER)
                        JOIN (
                            SELECT ROUTE_NUMBER, SUM(VALUE) AS AGE1_SEX1
                            FROM ROUTE_TABLE
                            WHERE AGE_RANGE_ID = 1 AND SEX_ID = 1
                            GROUP BY ROUTE_NUMBER
                        ) P11_DATA USING(ROUTE_NUMBER)
                        JOIN (
                            SELECT ROUTE_NUMBER, SUM(VALUE) AS AGE2_SEX1
                            FROM ROUTE_TABLE
                            WHERE AGE_RANGE_ID = 2 AND SEX_ID = 1
                            GROUP BY ROUTE_NUMBER
                        ) P21_DATA USING(ROUTE_NUMBER)
                        JOIN (
                            SELECT ROUTE_NUMBER, SUM(VALUE) AS AGE3_SEX1
                            FROM ROUTE_TABLE
                            WHERE AGE_RANGE_ID = 3 AND SEX_ID = 1
                            GROUP BY ROUTE_NUMBER
                        ) P31_DATA USING(ROUTE_NUMBER)
                    GROUP BY ROUTE_NAME, DEPART_ID, ARRIVE_ID, ROUTE_NUMBER, OPERATOR_ID, ON_DATE"""
        df = self.executeDataFrame(
            sql
        )
        df.columns = map(str.upper, df.columns)

        dti = DatetimeIndex(df.ON_DATE)

        df["ON_DAY"] = pandas.to_numeric(Series(dti.dayofyear), errors='coerce')

        if self.logger is not None: self.logger.info('Route ' + route_name + ' is loaded')

        return df

    def saveAnomalys(self, df):
        if self.logger is not None: self.logger.info('Saving results')
        #df.to_sql('ANOMALY_DETECTION', self.connection, if_exists='append')
        print(df)
        if self.logger is not None: self.logger.info('Anomalies is saved')