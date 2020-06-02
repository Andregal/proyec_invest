import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold


def import_data():
    # importar dataset
    data = pd.read_csv(r'C:\Users\full_\OneDrive\Escritorio\Tesis 2 Version 2\Libro2.csv')
    #data = data.drop("package_name")
    print (data.head)

    # conseguir lista de los nombres de las columnas
    headers = list(data.columns.values)

    # Separar entre variables dependientes e independientes
    x = data[headers[:-1]]
    y = data[headers[-1:]].values.ravel()

    return x, y

if __name__ == '__main__':
    # Conseguir training y testing sets
    x, y = import_data()

    # setear a 3 splits
    skf = StratifiedKFold(n_splits=3)

    # listas blancas donde almacenar variables predictoras y esperadas
    predicted_y = []
    expected_y = []

    # particion de la data
    for train_index, test_index in skf.split(x, y):
        # specific ".loc" syntax for working with dataframes
        x_train, x_test = x.loc[train_index], x.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # create and fit classifier
        classifier = GaussianNB()
        classifier.fit(x_train, y_train)

        # store result from classification
        predicted_y.extend(classifier.predict(x_test))

        # store expected result for this specific fold
        expected_y.extend(y_test)

    # save and print accuracy
    accuracy = metrics.accuracy_score(expected_y, predicted_y)
    print("Accuracy: " + accuracy.__str__())