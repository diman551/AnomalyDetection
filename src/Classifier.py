import pickle
import random
from collections import Counter

import numpy
import pandas
from sklearn.cluster import DBSCAN
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

class Classifier:
    def __init__(self, transport_type, connector, sample_path = None):
        if sample_path is None:
            clear_sample = self.__createClearSample(transport_type, connector)
            sample = self.__generateAnomalyData(clear_sample)
        else:
            sample = pandas.read_csv(sample_path, index_col=0)
            self.axesMax = numpy.max(sample, axis=0)[:-1]
            self.axesMin = numpy.min(sample, axis=0)[:-1]

        X = sample.loc[:, :'MAX(AGE3_SEX1)']
        y = sample[['Class']].values

        self.scaler = StandardScaler().fit(X)
        X_scaled = self.scaler.transform(X)

        self.classifier = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1).fit(X_scaled, y)

        y_normal = []
        for value in y:
            if value != 0:
                y_normal.append(-1)
            else:
                y_normal.append(0)

        self.accuracy = []

        kf = KFold(n_splits=5, shuffle=True)

        neigh = KNeighborsClassifier(n_neighbors=10)
        cvs = cross_val_score(X=X, y=y, estimator=neigh, cv=kf)
        self.accuracy.append(numpy.array(cvs).mean())

        neigh_n = KNeighborsClassifier(n_neighbors=10)
        cvs_n = cross_val_score(X=X, y=y_normal, estimator=neigh_n, cv=kf)
        self.accuracy.append(numpy.array(cvs_n).mean())

        sample.to_csv('./samples/transport_' + str(transport_type) + '.csv')
        self.__save(transport_type)

    @classmethod
    def load(cls, transport_type, connector, logger=None):
        try:
            if logger is not None: logger.info('Try to load classifier')
            classifier_dump = open('./classifiers/transport_'+str(transport_type)+'.classifier', 'rb')
            classifier = pickle.load(classifier_dump)
            if logger is not None: logger.info('Classifier is loaded')
            return classifier
        except Exception:
            if logger is not None: logger.info('Load classifier exception, try to create new classifier')
            return Classifier(transport_type, connector)

    def getAnomalys(self, data, route_nums, labels):
        for i in range(0, len(labels)):
            if labels[i] == 1:
                labels[i] = 0
            else:
                labels[i] = 1
        df = pandas.DataFrame(columns=['ROUTE_NUMBER'])
        for i in range(0, len(labels)):
            if labels[i] != 0:
                row = [
                    route_nums[i]
                ]
                df.loc[len(df)] = row
        return df

    def classify(self, data):
        data_scaled = self.scaler.transform(data)
        return self.classifier.predict(data_scaled)

    def __createClearSample(self, transport_type, connector):
        route_names = connector.getRouteNames(transport_type)
        sample_route_names = set()
        while len(sample_route_names) < 10:
            sample_route_names.add(random.choice(route_names))

        routes_df = []
        for route_name in sample_route_names:
            routes_df.append(self.__cleanOutData(route_name, 1.1, connector))

        return pandas.concat(routes_df)

    def __generateAnomalyData(self, clear_sample):
        df = clear_sample.copy(deep=True)
        df["Class"] = 0

        axesMax = numpy.max(df, axis=0)[:-1]
        axesMin = numpy.min(df, axis=0)[:-1]

        self.axesMax = axesMax
        self.axesMin = axesMin

        axesADCount = int(len(df) / (len(axesMax) - 1))  # не учитываем первый столбец

        for i in range(1, len(axesMax)):
            aMax = axesMax[i]
            aMin = axesMin[i]
            width = aMax - aMin

            for j in range(0, axesADCount):
                row = self.__generateNormalDataValues(axesMin, axesMax, i)
                if bool(random.getrandbits(1)):
                    space = int(aMin * 0.1)
                    highValue = aMin - space
                    row[i] = random.randint(0, highValue)
                else:
                    space = int(width * 0.1)
                    highValue = aMax + space * 2
                    row[i] = random.randint(aMax + space, highValue)
                df.loc[len(df)] = row

        return df

    def __cleanOutData(self, routeName, density, connector):
        df = connector.getRouteData(routeName)
        data_df = df[["ON_DAY", "SUM_VALUES",
                    "MAX(AGE1_SEX0)", "MAX(AGE2_SEX0)", "MAX(AGE3_SEX0)",
                    "MAX(AGE1_SEX1)", "MAX(AGE2_SEX1)", "MAX(AGE3_SEX1)", ]]
        route_nums = df[["ROUTE_NUMBER"]].values[:, 0]

        scaled_data = scale(data_df.values)
        anomalyDataRouteNums, labels = self.__clusterFilter(scaled_data, route_nums, density)
        return self.__cleanOut(data_df, labels)

    def __clusterFilter(self, data, route_nums, density):
        if len(data) != len(route_nums):
            raise Exception("Data size should be equal route_nums size")

        clustering = DBSCAN(eps=density, min_samples=1).fit(data)
        anomaly_routes = self.__getAnomalyRoutes(route_nums, clustering.labels_)
        return anomaly_routes, clustering.labels_

    def __cleanOut(self, data, labels):
        df = data.copy(deep=True)

        counter = Counter(labels.tolist())
        maxValue = 0
        maxKey = 0
        for key in counter:
            if counter[key] > maxValue:
                maxValue = counter[key]
                maxKey = key

        idxs = []
        for i in range(0, len(labels)):
            if labels[i] != maxKey:
                idxs.append(i)

        return df.drop(idxs)

    def __getAnomalyRoutes(self, route_nums, labels):
        anomaly_labels = self.__transformLabels(labels)
        anomaly_routes = []
        for i in range(0, len(route_nums)):
            if anomaly_labels[i] == -1:
                anomaly_routes.append(route_nums[i])

        return anomaly_routes

    def __transformLabels(self, labels):
        new_l = []
        for element in labels:
            if element >= 1:
                new_l.append(-1)
            else:
                new_l.append(element)
        return numpy.array(new_l)

    def __generateNormalDataValues(self, axesMin, axesMax, classNum):
        row = []
        for i in range(0, len(axesMax)):
            row.append(random.randint(axesMin[i], axesMax[i]))

        row.append(classNum)
        return row

    def __anomalyClass(self, aClass):
        switch = {
            1: "SUM VALUE",
            2: "AGE 1 SEX 0",
            3: "AGE 2 SEX 0",
            4: "AGE 3 SEX 0",
            5: "AGE 1 SEX 1",
            6: "AGE 2 SEX 1",
            7: "AGE 3 SEX 1"
        }
        return switch[aClass]

    def __save(self, transport_type):
        classifier_dump = open('./classifiers/transport_' + str(transport_type) + '.classifier', 'wb')
        pickle.dump(self, classifier_dump, pickle.HIGHEST_PROTOCOL)