import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle

data = pd.read_csv("buddhist_psychology_training_dataset.csv")

X = data[[
"childhood_support",
"stress_level",
"work_pressure",
"emotional_attachment",
"mental_clarity",
"coping_skill"
]]

y = data["personality_type"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train,y_train)

# save model
pickle.dump(model, open("buddhist_model.pkl","wb"))

print("Buddhist model saved!")