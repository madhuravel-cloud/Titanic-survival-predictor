import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
dat=pd.read_csv(r"titanic\train.csv")
d=pd.read_csv(r"titanic\test.csv")
data2=d.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
data1=dat.drop(['PassengerId','Name','Ticket','Cabin','Survived'],axis=1)
data1["Age"]=data1["Age"].fillna(data1["Age"].median())
data2["Age"]=data2["Age"].fillna(data1["Age"].median())
data1["Fare"]=data1["Fare"].fillna(data1["Fare"].median())
data2["Fare"]=data2["Fare"].fillna(data1["Fare"].median())
data1["Embarked"]=data1["Embarked"].fillna(data1["Embarked"].mode()[0])
data2["Embarked"]=data2["Embarked"].fillna(data1["Embarked"].mode()[0])
sex_encoder=LabelEncoder()
data1["Sex"]=sex_encoder.fit_transform(data1["Sex"])
data2["Sex"]=sex_encoder.transform(data2["Sex"])
embarked_encoder=LabelEncoder()
data1["Embarked"]=embarked_encoder.fit_transform(data1["Embarked"])
data2["Embarked"]=embarked_encoder.transform(data2["Embarked"])
model=RandomForestClassifier()
model.fit(data1,dat["Survived"])
y_pred=model.predict(data2)
df=pd.DataFrame({"PassengerId":d["PassengerId"],"Survived":y_pred})
df.to_csv("submission.csv",index=False)
