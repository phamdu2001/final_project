import pickle as pkl

with open("train.pkl", "rb") as f:
    X, y = pkl.load(f)
loaded_model = pkl.load(open("kneighbor_model.pickle", "rb"))
print(loaded_model.predict(X))
print(loaded_model.score(X,y))
