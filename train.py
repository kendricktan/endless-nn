import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier


INPUT = []
OUTPUT = []

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(64, 16), random_state=1, verbose=True)

DATA = ['jump_data.csv', 'normal_data.csv']
for fi in DATA:
    with open(fi, 'r') as f:
        df = pd.read_csv(f)

        # Read into dataframes
        for i in df.index:
            # Can't turn string to numpy array easily for some reason?
            INPUT.append(np.fromstring(df['input'][i][2:-1].replace('\n', '').replace('.', ''), sep=' '))
            OUTPUT.append(int(df['output'][i][1]))

INPUT = np.array(INPUT)
OUTPUT = np.array(OUTPUT)

clf.fit(INPUT, OUTPUT)

