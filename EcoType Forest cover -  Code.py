#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,LabelEncoder
from imblearn.over_sampling import SMOTE, SMOTENC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV


# In[2]:


df = pd.read_csv("cover_type (1).csv")


# In[3]:


data = pd.DataFrame(df)
df.info()


# In[4]:


data.head()


# ### Outliers

# In[5]:


numerical_cols = ['Elevation',
'Aspect',
'Slope',
'Horizontal_Distance_To_Hydrology',
'Vertical_Distance_To_Hydrology',
'Horizontal_Distance_To_Roadways',
'Hillshade_9am',
'Hillshade_Noon',
'Hillshade_3pm',
'Horizontal_Distance_To_Fire_Points']

plt.figure(figsize=(15, 7))
plt.suptitle("Boxplots after outliers Removal")
for i in range(0, len(numerical_cols)):
    plt.subplot(2, 6, i+1)
    sns.boxplot(y=data[numerical_cols[i]],color='purple')
    plt.tight_layout()


# In[6]:


for col in numerical_cols:

    Q1 = data[col].quantile(0.25)  # 25%
    Q3 = data[col].quantile(0.75)  # 75%
    IQR = Q3 - Q1 

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
    data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])


# In[7]:


Q1 = data['Elevation'].quantile(0.25)  # 25%
Q3 = data['Elevation'].quantile(0.75)  # 75%
IQR = Q3 - Q1  

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("Upper Bound:",upper_bound)
print("Lower Bound:",lower_bound)


# In[11]:


plt.figure(figsize=(15, 7))
plt.suptitle("Boxplots after outliers Removal")
for i in range(0, len(numerical_cols)):
    plt.subplot(2, 6, i+1)
    sns.boxplot(y=data[numerical_cols[i]],color='green')
    plt.tight_layout()


# In[12]:


data.isnull().sum()


# ### Skewness

# In[14]:


skewness = data[numerical_cols].skew()
skewness


# In[15]:


skewed_cols= skewness[skewness>0.7].index
skewed_cols


# In[16]:


data.head()


# ### Feature Engineering

# In[17]:


data['Distance_To_Hydrology']= np.sqrt(data['Horizontal_Distance_To_Hydrology']**2+ data['Vertical_Distance_To_Hydrology']**2)
data['Water_Elevation']= data['Elevation']-data['Vertical_Distance_To_Hydrology']
data['Area_Steepness'] = data['Slope'] / (data['Elevation'] + 1e-5)
data["Hillshade_Avg"]= (data["Hillshade_9am"] + data["Hillshade_Noon"] + data["Hillshade_3pm"]) / 3


# In[18]:


data['Wilderness']=data[[w for w in data.columns if "Wilderness" in w]].idxmax(axis=1)
data['Soil_Type']=data[[s for s in data.columns if "Soil_Type" in s]].idxmax(axis=1)


# In[19]:


data=data.drop(columns=[w for w in data.columns if "Wilderness_" in w])
data=data.drop(columns=[s for s in data.columns if "Soil_Type_" in s])


# In[20]:


# Target Encoding
target_encoder=LabelEncoder()
data['Cover_Type']=target_encoder.fit_transform(data['Cover_Type'])
with open('target_encoder.pkl', 'wb') as f:
    pickle.dump(target_encoder, f)


# In[21]:


data.info()


# ### Data Split

# In[22]:


X= data.drop(columns=['Cover_Type','Hillshade_9am','Hillshade_Noon','Hillshade_3pm'])
#Y= resampled_df['Cover_Type']
Y= data['Cover_Type']


# In[23]:


X.head()


# In[24]:


X.info()


# In[25]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(X,Y, test_size=0.2, random_state=42, stratify=Y)


# In[26]:


x_train.head()


# In[27]:


x_test.head()


# ### SMOTE

# In[28]:


y_train.value_counts()


# In[29]:


categorical_cols=['Wilderness','Soil_Type']
categorical_indices=[x_train.columns.get_loc(col) for col in categorical_cols]

smote=SMOTENC(categorical_features=categorical_indices,random_state=42)
x_res_train,y_res_train=smote.fit_resample(x_train,y_train)


# In[30]:


x_res_train


# In[31]:


y_res_train.value_counts()


# ### Encoding

# In[32]:


wilderness_encoder=LabelEncoder()
soil_encoder=LabelEncoder()

x_res_train['Wilderness']=wilderness_encoder.fit_transform(x_res_train['Wilderness'])
x_test['Wilderness']=wilderness_encoder.transform(x_test['Wilderness'])

with open('wilderness_encoder.pkl', 'wb') as f:
    pickle.dump(wilderness_encoder, f)

x_res_train['Soil_Type']=soil_encoder.fit_transform(x_res_train['Soil_Type'])
x_test['Soil_Type']=soil_encoder.transform(x_test['Soil_Type'])

with open('soil_encoder.pkl', 'wb') as f:
    pickle.dump(soil_encoder, f)


# In[33]:


x_test[['Soil_Type','Wilderness']].head()


# ### Feature selection

# In[34]:


x_res_train[x_res_train.duplicated(keep=False)].any()


# In[35]:


x=x_res_train
y=y_res_train

RF_model=RandomForestClassifier(n_estimators=200,random_state=42)
RF_model.fit(x,y)


# In[36]:


important_cols=pd.DataFrame({
    'features':x.columns,
    'importance':RF_model.feature_importances_
})
important_cols=important_cols.sort_values('importance', ascending= False)
important_cols


# In[37]:


plt.figure(figsize=(10,6))
sns.barplot(x=important_cols['importance'][:20], y=important_cols['features'][:20], color='purple', width=0.6)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Top 10 Important Features - Random Forest")
plt.show()


# ### Model Building

# In[38]:


models=[DecisionTreeClassifier(), RandomForestClassifier(), KNeighborsClassifier(), XGBClassifier(n_estimators= 200, eval_metric='logloss')]

for model in models:
    model.fit(x_res_train,y_res_train)

    y_pred=model.predict(x_test)

    accuracy= accuracy_score(y_pred,y_test)
    confusion= confusion_matrix(y_pred,y_test)
    classification= classification_report(y_pred,y_test)
    print(str(model),'Accuracy:', accuracy)
    print(str(model),'confusion_matrix:\n',confusion)
    print(str(model),'classification_report:\n',classification)


# ### Cross Validation

# In[40]:


from sklearn.model_selection import cross_val_score, StratifiedKFold

skf= StratifiedKFold(n_splits=5, shuffle= True, random_state=42)
for model in models:
    cv=cross_val_score(model,x_res_train,y_res_train,cv=skf,scoring='accuracy')
    avg=cv.mean()
    std=cv.std()
    print(f"{model}:AVG= {avg},STD= {std}")


# In[44]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
pipeline= Pipeline([
    ('scaler',StandardScaler()),
    ('logreg',LogisticRegression(max_iter=1000,solver='saga', C=0.5))
])

pipeline.fit(x_res_train,y_res_train)
y_pred=pipeline.predict(x_test)
accuracy= accuracy_score(y_pred,y_test)
confusion= confusion_matrix(y_pred,y_test)
classification= classification_report(y_pred,y_test)

print(f"LogisticRegression Accuracy: {accuracy}")
print(f"LogisticRegression confusion_matrix:\n{confusion}")
print(f"LogisticRegression classification_report:\n{classification}")


# ### Cross Validation

# In[46]:


from sklearn.model_selection import cross_val_score, StratifiedKFold

skf= StratifiedKFold(n_splits=5, shuffle= True, random_state=42)
for model in models:
    cv=cross_val_score(model,x_res_train,y_res_train,cv=skf,scoring='accuracy')
    avg=cv.mean()
    std=cv.std()
    print(f"{model}:AVG= {avg},STD= {std}")


# In[48]:


#pipeline= Pipeline([
#    ('scaler',StandardScaler()),
#    ('logreg',LogisticRegression(max_iter=1000,solver='saga', C=0.5, class_weight='balanced'))
#])
cv=cross_val_score(pipeline,x_res_train,y_res_train,cv=skf,scoring='accuracy')
avg=cv.mean()
std=cv.std()
print(f"LogisticRegression:AVG= {avg},STD= {std}")


# ### Hyperparameter Tuning

# In[62]:


from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

subset_size = 200000  
X_sub = x_res_train.sample(n=subset_size, random_state=42)
y_sub = y_res_train.loc[X_sub.index]

rf_para_grid={
    'n_estimators':[100,200],
    'max_depth':[10,15],
    'min_samples_split':[5,10],
    'min_samples_leaf':[2,4],
    'max_features':['sqrt','log2'],
    'bootstrap':[True]
}

grid= RandomizedSearchCV(RandomForestClassifier(random_state=42), rf_para_grid, n_iter=15, cv=3, scoring='accuracy', n_jobs=-1, verbose=1,random_state=42, return_train_score=True)
grid.fit(X_sub,y_sub)


# In[54]:


X_sub = x_res_train.sample(n=200000, random_state=42)
y_sub = y_res_train.loc[X_sub.index]

rf_para_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
    'max_features': ['sqrt']
}

grid = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_para_grid,
    n_iter=15,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

grid.fit(X_sub, y_sub)


# In[63]:


random=grid
best_model=random.best_estimator_


# In[64]:


print(f"best parameter=",random.best_params_)
print(f"best accuracy=",random.best_score_)


# In[65]:


with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)


# In[ ]:




