#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score


# In[2]:


df = pd.read_csv("Treasury.csv", sep = ',')


# In[3]:


df


# # KNN

# In[4]:


df = df.drop(df.columns[0:2], axis=1)
df_x_knn = df.iloc[:,6:8]
df_y_knn = df.iloc[:,-1]


# In[5]:


df


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(df_x_knn, df_y_knn, test_size=0.3,
                 random_state=1, stratify=df_y_knn)


# In[9]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train_knn)
X_train_std = sc.transform(X_train_knn)
X_test_std = sc.transform(X_test_knn)


# In[10]:


from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, test_idx = None,
                          resolution = 0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
       # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                              np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                       alpha=0.8, c=colors[idx],
                       marker=markers[idx], label=cl,
                       edgecolor='black')
       # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                       c='', edgecolor='black', alpha=1.0,
                       linewidth=1, marker='o',
                       s=100, label='test set')


# In[11]:


from sklearn.neighbors import KNeighborsClassifier

k_range = range(1,26)
score = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors= k)
    knn.fit(X_train_std, y_train_knn)
    y_pred = knn.predict(X_test_std)
    score.append(accuracy_score(y_test_knn, y_pred))


# In[12]:


best_k = score.index(np.max(score)) + 1


# In[13]:


best_k


# In[14]:


score


# In[17]:


knn_1 = KNeighborsClassifier(n_neighbors=best_k)
knn_1.fit(X_train_std, y_train_knn)

X_combined_std = np.vstack((X_train_std, X_test_std)) 
y_combined = np.hstack((y_train_knn, y_test_knn))
plot_decision_regions(X_combined_std, y_combined, classifier=knn_1, test_idx=range(105,150))
plt.xlabel('ctd1_percent')
plt.ylabel('delivery_cost')
plt.legend(loc='upper left')
plt.show()


# # Decision Tree

# In[22]:


from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
df_x_tree = df.iloc[:,7:9]
df_y_tree = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(df_x_tree, df_y_tree, test_size=0.3, random_state=1, stratify=df_y_tree)
sc = StandardScaler()
sc.fit(X_train)
X_train_std_tree = sc.transform(X_train)
X_test_std_tree = sc.transform(X_test)
tree = DecisionTreeClassifier(criterion='gini',max_depth=4, random_state=1)
tree.fit(X_train_std_tree, y_train)

X_combined = np.vstack((X_train_std_tree, X_test_std_tree)) 
y_combined = np.hstack((y_train, y_test)) 
plot_decision_regions(X_combined ,y_combined, classifier=tree, test_idx=range(105, 150))
plt.xlabel('delivery_cost') 
plt.ylabel('delivery_ratio') 
plt.legend(loc='upper left') 
plt.show()


# In[20]:


print("My name is {Zhuoyuan Zhang}")
print("My NetID is: {zz10}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


# In[ ]:




