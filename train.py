import cPickle

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

INPUT = []
OUTPUT = []

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(64, 16), random_state=1, verbose=True)

DATA = ['data/output_0.csv', ]  # , 'data/output_1.csv']
for fi in DATA:
    with open(fi, 'r') as f:
        df = pd.read_csv(f)

        # Read into dataframes
        for i in df.index:
            # Can't turn string to numpy array easily for some reason?
            INPUT.append(np.fromstring(df['input'][i][2:-1].replace('\n', '').replace('.', ''), sep=' '))
            OUTPUT.append(int(df['output'][i]))

INPUT = np.array(INPUT)
OUTPUT = np.array(OUTPUT)

clf.fit(INPUT, OUTPUT)

with open('model.pkl', 'wb') as fid:
    cPickle.dump(clf, fid)
