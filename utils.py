from sklearn import svm, metrics
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from math import ceil

# compute accuracy of given featureset with svm and 5-fold cross validation
def SvmAccuracy(feature, dataset):
    # find selected features
    features = []
    for item in range(len(feature)):
        if feature[item] == 1:
            features.append(item)
    # transform data to only contain given featureset
    dataset = dataset[features]
    # split data and labels
    data = dataset.iloc[:,:-1]
    labels = dataset.iloc[:,-1]
    # create classifier object
    classifier = svm.SVC(decision_function_shape="ovo")
    # cross validate and get accuracy score for each fold
    scores = cross_val_score(classifier, data, labels, cv=5, scoring='accuracy')
    # compute average accuracy
    average_score = 0
    for item in scores:
        average_score += item
    average_score /= len(scores)
    return average_score
# print best,average and worst member of each population/iteration for given model
def PrintResults(model):
    for i in range(len(model.best_of_generation)):
        print(model.best_of_generation[i], model.average_of_generation[i], model.worst_of_generation[i])

# compute gain ratio using tree classificatio and return best split percentage of features
def ImportantFeatures(dataset, split):
    # split data and labels
    data = dataset.iloc[:,:-1]
    labels = dataset.iloc[:,-1]
    # declare and fit tree classifier
    classifier = DecisionTreeClassifier(criterion = 'entropy')
    classifier.fit(data, labels)
    # get gain ratio of each feature
    importances = classifier.feature_importances_
    # transform list so it includes both feature name and score
    temp = []
    for i in range(len(importances)):
        temp.append((i,importances[i]))
    # sorting features by importance
    temp.sort(key=lambda p: p[1], reverse = True)
    # picking specified split of features
    temp = temp[:ceil((dataset.shape[1]-1)*split)]
    result = []
    for i in range(len(temp)):
        result.append(temp[i][0])
    result.append(dataset.shape[1]-1)
    result.sort()
    return result

# converts each dataset to required format
def PrepareDataset(selected_dataset):
    if selected_dataset == 'wine':
        df1 = pd.read_csv("dataset/wine.data", header = None)
        c = df1.columns[1:]
        df2 = df1[c]
        df2[0] = df1.iloc[:][0]
        df2.columns = range(df2.shape[1])
        df2.sample(frac=1).reset_index(drop=True)
    elif selected_dataset == 'ionosphere':
        df2 = pd.read_csv("dataset/ionosphere.data", header = None)
        df2.sample(frac=1).reset_index(drop=True)
    elif selected_dataset == 'musk1':
        df2 = pd.read_csv("dataset/musk1.csv", header = None)
        df2 = df2[range(2,df2.shape[1])]
        df2.columns = range(df2.shape[1])
        df2 = df2.sample(frac=1).reset_index(drop=True)
    elif selected_dataset == 'WBDC':
        df1 = pd.read_csv("dataset/wdbc.data", header = None)
        c = df1.columns[2:]
        df2 = df1[c]
        df2[0] = df1.iloc[:][1]
        df2.columns = range(df2.shape[1])
        df2.sample(frac=1).reset_index(drop=True)
    elif selected_dataset == 'german':
        df2 = pd.read_csv("dataset/german.data-numeric", header = None)
        df2.sample(frac=1).reset_index(drop=True)
    elif selected_dataset == 'lung':
        df1 = pd.read_csv("dataset/lung-cancer.data", header = None)
        c = df1.columns[1:]
        df2 = df1[c]
        df2[0] = df1.iloc[:][0]
        df2.columns = range(df2.shape[1])
        df2.sample(frac=1).reset_index(drop=True)
    elif selected_dataset == 'Hill_Valley':
        df2 = pd.read_csv("dataset/Hill_Valley_without_noise_Training.data", header = 0)
        df2.columns = range(df2.shape[1])
        df2.sample(frac=1).reset_index(drop=True)
    return df2
