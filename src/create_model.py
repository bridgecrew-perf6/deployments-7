import sklearn
import numpy as np
import joblib
import xgboost as xgb
import os

FILE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir))
MODELS_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'models'))
MODEL_FILE = os.path.join(MODELS_DIR, "model.pkl")

from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=108, stratify=y)

def train():
	bt = xgb.XGBClassifier(max_depth=5, learning_rate=0.2, n_estimators=10, objective='multi:softmax')
	bt.fit(X_train, y_train, verbose=False)
	return bt

def predict():
	bt=joblib.load(MODEL_FILE)
	print("REAL: ", y_test)
	print("PREDICTED: ", bt.predict(X_test))

def main():
	bt=train()
	joblib.dump(bt, MODEL_FILE)
	predict()

if __name__== "__main__":
		main()
