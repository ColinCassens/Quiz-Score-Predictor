from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GM
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np
import math
import matplotlib.pyplot as plt


def filter1(userdict):
    returndict = {}
    for item in userdict.items():
        if len(item[1]) >= 5:
            returndict[item[0]] = item[1]
    return returndict


def filter2(userdict):
    returndict = {}
    for item in userdict.items():
        if len(item[1]) >= 47:
            returndict[item[0]] = item[1]
    return returndict


def clusteringTest(data):
    # Calculate the average of all data, by student.
    print("\nClustering Test:\n")
    avgdata = []
    for student in data.keys():
        sdata = data[student]
        list1 = sdata[0]
        avg = [float(list1[i]) for i in range(len(list1))]
        for index in range(1, len(sdata)):
            for i in range(len(avg)):
                avg[i] += float(sdata[index][i])
        for i in range(len(avg)):
            avg[i] = avg[i] / len(sdata)
        avg.remove(avg[7])
        avg.remove(avg[9])
        avgdata.append(avg)

    # Run K-means
    kmeans = KMeans(n_clusters=8).fit(avgdata)
    # print(kmeans.cluster_centers_)

    # Run GMM
    GMM = GM(n_components=2, covariance_type='tied').fit(avgdata)

    # Calculate MSE
    print('Kmeans MSE = ' + str(kmeans.inertia_ / len(avgdata)))
    cov = GMM.covariances_
    MSE = [np.average(np.diag(cov[x])) for x in range(len(cov))]
    print('GMM MSE using TIED covariance mode = ' + str(np.average(MSE)))


def avgScorePredictor(data):
    print("\nAvg Score Predictor:\n* Trained using students who have completed at least 50% of quizzes")
    # Get the avg feature data and quiz score for each student
    avgdata = []
    avgscore = []
    for student in data.keys():
        sdata = data[student]
        list1 = sdata[0]
        avg = [float(list1[i]) for i in range(len(list1))]
        for index in range(1, len(sdata)):
            for i in range(len(avg)):
                avg[i] += float(sdata[index][i])
        for i in range(len(avg)):
            avg[i] = avg[i] / len(sdata)
        avg.remove(avg[7])
        avgscore.append(avg[9])
        avg.remove(avg[9])
        avgdata.append(avg)

    # divide the data into a training and a learning set
    trainingX = avgdata[0: int(0.8 * len(avgdata))]
    testX = avgdata[int(0.8 * len(avgdata)): len(avgdata)]
    trainingY = avgscore[0: int(0.8 * len(avgscore))]
    testY = avgscore[int(0.8 * len(avgscore)): len(avgscore)]

    # Run linear regression
    linearmodel = LinearRegression(fit_intercept=True)
    linearmodel.fit(trainingX, trainingY)

    # Run Ridge Regression on the data
    ridgemodel = Ridge(fit_intercept=True, alpha=0.9)
    ridgemodel.fit(trainingX, trainingY)

    # Determine which method produces better results
    print('MSE Linear Regression: ')
    print(error(testX, testY, linearmodel))
    print('RMSE Linear Regression: ')
    print(math.sqrt(error(testX, testY, linearmodel)))

    print('\nMSE Ridge Regression: ')
    print(error(testX, testY, ridgemodel))
    print('RMSE Ridge Regression: ')
    print(math.sqrt(error(testX, testY, ridgemodel)))


def error(X, y, model):
    return mean_squared_error(model.predict(X), y)


def individualScorePredictor(userdict):
    print('\nScore Predictor for Individual Quizzes:\n')
    data = []
    scores = []
    for student in userdict.keys():
        if student != 'userID':
            for vid in userdict[student]:
                v = [float(vid[i]) for i in range(len(vid))]
                v.remove(v[7])
                scores.append(v[9])
                v.remove(v[9])
                data.append(v)

    trainingX = data[0: int(0.8 * len(data))]
    testX = data[int(0.8 * len(data)): len(data)]
    trainingY = scores[0: int(0.8 * len(scores))]
    testY = scores[int(0.8 * len(scores)): len(scores)]

    # Run linear regression
    linearmodel = LinearRegression(fit_intercept=True)
    linearmodel.fit(trainingX, trainingY)
    linearpredict = linearmodel.predict(testX)

    # Run Ridge Regression on the data
    ridgemodel = Ridge(fit_intercept=True, alpha=0.2)
    ridgemodel.fit(trainingX, trainingY)
    ridgepredictor = ridgemodel.predict(testX)

    minmselin = findminMSE(linearpredict, testY)
    minmseridge = findminMSE(ridgepredictor, testY)

    # Determine which method produces better results
    print('MSE Linear Regression: ')
    print(minmselin)

    print('\nMSE Ridge Regression: ')
    print(minmseridge)


def findminMSE(predictor, test):
    min = 1
    # Implement Decision boundary
    for boundary in np.arange(0.3, 0.7, 0.001):
        output = []
        for index in range(len(predictor)):
            if predictor[index] >= boundary:
                output.append(1)
            else:
                output.append(0)
        mse = mean_squared_error(output, test)
        if mse < min:
            min = mse
    return min


if __name__ == "__main__":
    file = open('behavior-performance.txt')

    # Read the input file into UserDict
    # UserDict 'userID': ['VidID', 'fracSpent', 'fracComp', 'fracPlayed', 'fracPaused', 'numPauses', 'avgPBR', 'stdPBR', 'numRWs', 'numFFs', 's']
    userdict = {}
    for line in file:
        line1 = line.split()
        if line1[0] in userdict.keys():
            userdict[line1[0]].append([line1[x] for x in range(1, len(line.split()))])
        else:
            userdict[line1[0]] = [[line1[x] for x in range(1, len(line.split()))]]

    # Filter the data set to only have students who have completed 5 videos
    filter1Data = filter1(userdict)

    # Test how well the data can be clustered using K-means and GMM
    clusteringTest(filter1Data)

    '''
    Run Linear and Ridge Regression to determine if we can predict a students
        average score on all quizes based on video watching behavior
        NOTE: Only using students who have completed 50% of the quizzes
    '''
    filter2Data = filter2(userdict)
    avgScorePredictor(filter2Data)

    '''
    Run Linear and Ridge Regression to determine if we can predict a students
        score on a particular video quiz based on the watching behavior
        for that particular video
    '''
    individualScorePredictor(userdict)

    print("\nNOTE: All regression trainers were trained with 80% of the\n   described data set and tested with the other 20%\n")
