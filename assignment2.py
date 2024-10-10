# %% [markdown]
# ### Imports

import numpy as np
# %%
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# %% [markdown]
# ### Load data

# %%
df = pd.read_csv('https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv')
df.head()

# %% [markdown]
# ### Create model

# %%
features_to_exclude = [
    'meal',
    'id',
    'DateTime',
]

y = df['meal']
x = df.drop(features_to_exclude, axis=1)

# %%
# Randomly sample our data --> 70% to train with, and 30% for testing
# x, xt, y, yt = train_test_split(x, y, test_size=0.3)

# %%
# Create the model and fit it using the train data
model = RF(n_estimators=100, n_jobs=-1, max_depth=70)
modelFit = model.fit(x, y)

# %% [markdown]
# ### Make predictions

# %%
test_data = pd.read_csv('https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv')
xt = test_data.drop(features_to_exclude, axis=1)

# %%
pred = modelFit.predict(xt)
np.unique(pred)


