import numpy as np;
import plotly.express as px;
import pandas as pd;
from sklearn.metrics import confusion_matrix;
from sklearn.metrics import accuracy_score;

data = pd.read_csv('data.csv');
velocity = data['Velocity'].tolist();
escaped = data['Escaped'].tolist();

m, c = np.poly1d(np.polyfit(velocity, escaped, 1));

Y = [];
for x in velocity:
    y_value = m * x + c;
    if y_value < 0.5:
        Y.append(0);
    else:
        Y.append(1);

cm = confusion_matrix(escaped, Y);

accuracy = accuracy_score(escaped, Y);

print(cm);
print('The accuracy of the model is :- ' + str(accuracy));

