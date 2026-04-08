import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle

data = pd.read_csv("western_psychology_training_dataset.csv")

X = data[[
"openness",
"conscientiousness",
"extraversion",
"agreeableness",
"neuroticism",
"stress_level",
"coping_skill"
]]

y = data["personality_type"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train,y_train)

pickle.dump(model, open("western_model.pkl","wb"))

print("Western model saved!")