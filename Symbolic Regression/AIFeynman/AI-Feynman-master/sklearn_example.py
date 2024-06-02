from aifeynman import AIFeynmanRegressor

import pandas as pd
import pdb
df = pd.read_csv('example_data/example1.txt',sep=' ', header=None,
        index_col=False).dropna(how='all',axis=1)
print(df)
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
print('X:\n',X, '\ny:\n',y)
assert(X.shape[0] == len(y))

print('init')
est = AIFeynmanRegressor(
        BF_try_time=10,
        polyfit_deg=3,
        NN_epochs=100,
        # max_time=7*60*60
        max_time=60
        )

print('fit')
est.fit(X,y)

y_pred = est.predict(X)

print('y_pred:',y_pred)

print('self.best_model_:',est.best_model_)
print('complexity:',est.complexity())

