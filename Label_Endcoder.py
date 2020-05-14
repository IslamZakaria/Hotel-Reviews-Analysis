from sklearn.preprocessing import LabelEncoder
import pickle
def Feature_Encoder(X,cols):


    for c in cols:
        #lbl = LabelEncoder()
        #lbl.fit(list(X[c].values))
        lbl=pickle.load(open(c+".csv",'rb'))
        X[c] = lbl.transform(list(X[c].values))

    return X
