#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train['train_test'] = 1
test['train_test'] = 0

test['Survived'] = np.NaN 
full_dataset = pd.concat([train,test])

get_ipython().run_line_magic('matplotlib', 'inline')
full_dataset.columns


# In[5]:


full_dataset.info()


# In[6]:


full_dataset.describe()


# In[7]:


df_num = train[['Age','SibSp','Parch','Fare']]
df_cat = train[['Survived','Pclass','Sex','Ticket','Cabin','Embarked']]


# In[8]:


for i in df_num.columns:
    plt.hist(df_num[i])
    plt.title(i)
    plt.show()


# In[9]:


print(df_num.corr())
sns.heatmap(df_num.corr(),annot=True)


# In[10]:


pd.pivot_table(train,index='Survived',values= ['Age','SibSp','Parch','Fare'])


# In[11]:


for i in df_cat.columns:
    sns.barplot(df_cat[i].value_counts().index,df_cat[i].value_counts()).set_title(i)
    plt.show()


# In[12]:


print(pd.pivot_table(train,index = "Survived",columns ="Pclass",values ="Ticket",aggfunc = 'count'))
print()
print(pd.pivot_table(train,index = "Survived",columns ="Sex",values ="Ticket",aggfunc = 'count'))
print()
print(pd.pivot_table(train,index = "Survived",columns ="Embarked",values ="Ticket",aggfunc = 'count'))
print()


# In[13]:


df_cat.Cabin
#creating a new column 
train['cabin_multiple'] = train['Cabin'].apply(lambda x:0 if pd.isna(x) else len(x.split(' ')) )
train['cabin_multiple'].value_counts()


# In[14]:


pd.pivot_table(train,index="Survived",columns = 'cabin_multiple',values= 'Ticket',aggfunc ='count')


# In[15]:


train['cabin_adv'] = train['Cabin'].apply(lambda x: str(x)[0])


# In[16]:


print(train['cabin_adv'])
pd.pivot_table(train,index ="Survived",columns ='cabin_adv',values ='Ticket',aggfunc ='count')


# In[17]:


#understanding ticket values a bit better

train['numeric_ticket'] = train['Ticket'].apply(lambda x:1 if x.isnumeric() else 0)


# In[18]:


train['ticket_letters'] = train['Ticket'].apply(lambda x:''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)


# In[19]:


train['numeric_ticket'].value_counts()


# In[20]:


train['ticket_letters'].value_counts()


# In[21]:


#difference in numeric vs non-numeric tickets in survival rate
pd.pivot_table(train,index="Survived",columns = 'numeric_ticket',values ='Ticket',aggfunc = 'count')


# In[22]:


#survival rate across different tyicket types 
pd.pivot_table(train,index ='Survived',columns ='ticket_letters',values ='Ticket',aggfunc ='count')


# In[23]:


train['Name'].head(50)
train['name_title'] = train['Name'].apply(lambda x:x.split(',')[1].split('.')[0])
train['name_title'].value_counts()


# In[24]:


#create all categorical variables that we did above for both training and test sets 
full_dataset['cabin_multiple'] = full_dataset.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
full_dataset['cabin_adv'] = full_dataset.Cabin.apply(lambda x: str(x)[0])
full_dataset['numeric_ticket'] = full_dataset.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
full_dataset['ticket_letters'] = full_dataset.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)
full_dataset['name_title'] = full_dataset.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())

# replace nulls for countinuous data
full_dataset['Age'] =full_dataset['Age'].fillna(train['Age'].median())
full_dataset['Fare'] =full_dataset['Fare'].fillna(train['Fare'].median())

#dropping the embarked null data(only 2 rows)
full_dataset['Embarked'].dropna()

#normalization of the data so that the columns with greater value will not dominate the result 


#normalising the fare column
full_dataset['norm_fare'] = np.log(full_dataset['Fare']+1)
full_dataset['norm_fare'].hist()

# converting the pclass into a string datatype
full_dataset['Pclass'] = full_dataset['Pclass'].astype(str)

#creating dummy variables to store the data in categorical values
all_dummies = pd.get_dummies(full_dataset[['Pclass','Sex','Age','SibSp','Parch','norm_fare',
                                       'Embarked','cabin_adv','cabin_multiple','numeric_ticket','name_title','train_test']])

#split to train test again
X_train = all_dummies[all_dummies['train_test'] == 1].drop(['train_test'],axis =1)
X_test = all_dummies[all_dummies['train_test'] == 0].drop(['train_test'],axis =1)

y_train = full_dataset[full_dataset['train_test'] ==1]['Survived']
y_train.shape


# In[25]:


#scaling the data
#scaling is used to fit the data to a certain scale like 0-100 or 0-1, scaling is used when you are using techniques that
# are based on measuring how far the data points are like SVM or KNN
from sklearn.preprocessing import StandardScaler 
scale =StandardScaler()
all_dummies_scaled = all_dummies.copy()
#we use copy to make sure that the original dataset remains unchanged
all_dummies_scaled[['Age','SibSp','Parch','norm_fare']] = scale.fit_transform(all_dummies_scaled[['Age','SibSp','Parch','norm_fare']])
all_dummies_scaled

X_train_scaled = all_dummies_scaled[all_dummies_scaled['train_test'] == 1].drop(['train_test'],axis = 1)
X_test_scaled = all_dummies_scaled[all_dummies_scaled['train_test'] == 0].drop(['train_test'],axis = 1)

y_train =full_dataset[full_dataset['train_test']== 1]['Survived']


# In[26]:


from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# In[27]:


#Naive Bayes
gnb = GaussianNB()
cv = cross_val_score(gnb,X_train_scaled,y_train,cv = 5)
print(cv)
cv.mean()


# In[28]:


#Logistic Regression
lr =LogisticRegression(max_iter = 2000)
cv = cross_val_score(lr,X_train_scaled,y_train,cv =5)
print(cv)
print(cv.mean())


# In[29]:


#Decision Tree Classifier
dt = tree.DecisionTreeClassifier(random_state = 1)
cv = cross_val_score(dt,X_train,y_train,cv =5)
print(cv)
print(cv.mean())


# In[30]:


#Decision Tree Classifier
dt = tree.DecisionTreeClassifier(random_state = 1)
cv = cross_val_score(dt,X_train_scaled,y_train,cv =5)
print(cv)
print(cv.mean())


# In[31]:


#K Nearest Neighbours
knn = KNeighborsClassifier()
cv = cross_val_score(knn,X_train,y_train,cv =5)
print(cv)
print(cv.mean())


# In[32]:


#K Nearest Neighbours
knn = KNeighborsClassifier()
cv = cross_val_score(knn,X_train_scaled,y_train,cv =5)
print(cv)
print(cv.mean())


# In[33]:


#Random Forest Classifier
rf= RandomForestClassifier(random_state =1)
cv = cross_val_score(rf,X_train,y_train,cv =5)
print(cv)
print(cv.mean())


# In[34]:


#Random Forest Classifier
rf= RandomForestClassifier(random_state =1)
cv = cross_val_score(rf,X_train_scaled,y_train,cv =5)
print(cv)
print(cv.mean())


# In[35]:


#Support Vector Machine
svc= SVC(probability =True)
cv = cross_val_score(svc,X_train_scaled,y_train,cv =5)
print(cv)
print(cv.mean())


# In[36]:


from xgboost import XGBClassifier
xgb = XGBClassifier(random_state =1)
cv = cross_val_score(xgb,X_train_scaled,y_train,cv=5)
print(cv)
print(cv.mean())


# In[37]:


from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(estimators = [('lr',lr),('knn',knn),('rf',rf),('gnb',gnb),('svc',svc),('xgb',xgb)], voting = 'soft') 
cv = cross_val_score(voting_clf,X_train_scaled,y_train,cv=5)
print(cv)
print(cv.mean())


# In[38]:


voting_clf.fit(X_train_scaled,y_train)
y_hat_base_vc = voting_clf.predict(X_test_scaled).astype(int)
basic_submission = {'PassengerId': test.PassengerId, 'Survived': y_hat_base_vc}
base_submission = pd.DataFrame(data=basic_submission)
base_submission.to_csv('base_submission.csv', index=False)


# In[43]:


#hard voting classifier of 3 estimators
voting_hard = VotingClassifier(estimators =[('knn',knn),('rf',rf),('svc',svc)], voting = 'hard')

#soft voting classifier of 3 estimators
voting_soft = VotingClassifier(estimators =[('knn',knn),('rf',rf),('svc',svc)], voting = 'soft')

#voting classifier for percentage greater than 80 %
voting_all = VotingClassifier(estimators = [('knn',knn),('rf',rf),('svc',svc), ('lr', lr)], voting = 'soft') 

#voting classifier including XGB
voting_xgb = VotingClassifier(estimators = [('knn',knn),('rf',rf),('svc',svc), ('xgb', xgb),('lr', lr)], voting = 'soft')


# In[44]:


#predictions
voting_hard.fit(X_train_scaled,y_train)
voting_soft.fit(X_train_scaled,y_train)
voting_all.fit(X_train_scaled,y_train)
voting_xgb.fit(X_train_scaled,y_train)

rf.fit(X_train_scaled,y_train)
y_hard = voting_hard.predict(X_test_scaled).astype(int)
y_rf = rf.predict(X_test_scaled).astype(int)
y_soft = voting_soft.predict(X_test_scaled).astype(int)
y_all = voting_soft.predict(X_test_scaled).astype(int)
y_xgb = voting_xgb.predict(X_test_scaled).astype(int)


# In[51]:


# convert output to dataframe
final_data = {'PassengerId':test['PassengerId'], 'Survived':y_rf}
submission = pd.DataFrame(data = final_data)

final_data_1 = {'PassengerId':test['PassengerId'], 'Survived':y_hard}
submission_1 = pd.DataFrame(data = final_data_1)

final_data_2 = {'PassengerId':test['PassengerId'], 'Survived':y_soft}
submission_2 = pd.DataFrame(data = final_data_2)

final_data_3 = {'PassengerId':test['PassengerId'], 'Survived':y_all}
submission_3 = pd.DataFrame(data = final_data_3)

final_data_4 = {'PassengerId':test['PassengerId'], 'Survived':y_xgb}
submission_4 = pd.DataFrame(data = final_data_4)

final_data_composition =  {'PassengerId':test['PassengerId'], 'Survived_rf':y_rf,'Survived_hard':y_hard,
                           'Survived_soft':y_soft,'Survived_all':y_all}
comparison = pd.DataFrame(data = final_data_composition)


# In[53]:


#track differences between outputs 
comparison['difference_rf_vc_hard'] = comparison.apply(lambda x: 1 if x.Survived_hard != x.Survived_rf else 0, axis =1)
comparison['difference_soft_hard'] = comparison.apply(lambda x: 1 if x.Survived_hard != x.Survived_soft else 0, axis =1)
comparison['difference_hard_all'] = comparison.apply(lambda x: 1 if x.Survived_all != x.Survived_hard else 0, axis =1)
comparison['difference_hard_all'].value_counts()


# In[54]:


submission.to_csv('submission_rf.csv', index =False)
submission_1.to_csv('submission_hard.csv',index=False)
submission_2.to_csv('submission_soft.csv', index=False)
submission_3.to_csv('submission_all.csv', index=False)
submission_4.to_csv('submission_xgb.csv', index=False)


# In[ ]:


#Credits:
#this notebook was made possible by KenJee's Notebook on the same

