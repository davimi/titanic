import pandas as pd
from sklearn import linear_model
import pickle


def train_model():
    # load data
    train_data = pd.read_csv('../resources/trainingdata.csv')

    #prepare for training
    x_train = train_data[['Pclass', 'Sex', 'Age', 'SibSp',
                     'Parch', 'Fare']]

    # data prep
    x_train['Sex'] = x_train.Sex.map(lambda s: 1 if s == 'female' else 0)
    mean_age = x_train.Age.mean()
    x_train['Age'] = x_train.Age.map(lambda a: mean_age if pd.isna(a) else a)

    y_train = train_data.Survived

    # model training
    model = linear_model.LinearRegression(normalize=True)
    model.fit(x_train, y_train)

    # export
    pickle.dump(model, open('../resources/model', 'wb'))

if __name__ == "__main__":
    train_model()