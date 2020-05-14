from sklearn.preprocessing import LabelEncoder
import pickle
def Feature_Encoder_try(X,cols):


    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        pickle.dump(lbl,open(c + ".csv", 'wb'))
        X[c] = lbl.transform(list(X[c].values))

    return X
