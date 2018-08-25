
# coding: utf-8

# In[46]:


import numpy as np
import math 
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report,accuracy_score


# In[139]:


def AutoSVC(clf_data,search_times=3,cv=3,step=2,n_jobs=4):
    '''
    AutoSVC is a function,useful and user-friendly,
    helps workers to do machine learning experiments.
    See runAutoSVC for User Guide.
    
    Author: Pacoo
    Time: 2018-08-25
    '''
    # train and test data
    train_features = clf_data['train_features']
    train_labels = clf_data['train_labels']
    train_labels = train_labels.ravel() 
    test_features = clf_data['test_features']
    test_labels = clf_data['test_labels']
    test_labels = test_labels.ravel()
    
    # deep grid search
    C_start = -14
    C_stop = 15
    gamma_start = -14
    gamma_stop = 15
    for i in range(1,search_times+1):
        tuned_parameters = [
            {
                'kernel':['rbf'],
                'C':[math.pow(2,x) for x in np.arange(C_start,C_stop,step)],
                'gamma':[math.pow(2,x) for x in np.arange(gamma_start,gamma_stop,step)]
            }
        ]
        # GridSearchCrossValidate
        clf = GridSearchCV(SVC(), tuned_parameters,cv=cv,n_jobs=n_jobs)
        clf.fit(train_features, train_labels)
        
        best_C = math.log2(clf.best_params_['C'])
        best_g = math.log2(clf.best_params_['gamma'])
        print('%d th search:C = %.4f,g = %.4f,score = %.3f'
                 %(i,clf.best_params_['C'],clf.best_params_['gamma'],clf.best_score_))
        C_start = best_C - 2.25*step
        C_stop = best_C +2.26*step
        gamma_start = best_g - 2.25*step
        gamma_stop = best_g + 2.26*step
        step = 0.75*step
    
    # prediction and report
    y_true,y_pred = test_labels,clf.predict(test_features)
    clf_report = classification_report(y_true,y_pred)
    conf_matrix = np.zeros((len(np.unique(y_true)),len(np.unique(y_true))))
    misclf_index = []
    for i in range(0,len(y_true)-1):
        if not(y_true[i] == y_pred[i]):
            misclf_index.append(i)
        conf_matrix[y_true[i]-1][y_pred[i]-1] += 1
    acc = accuracy_score(y_true,y_pred)
    print('accuracy:%.4f'%acc)
    
    # return results
    automan = {
        'clf':clf,
        'y_t':y_true,
        'y_p':y_pred,
        'acc':acc,
        'clf_rep':clf_report,
        'conf_mat':conf_matrix,
        'misclf_index':misclf_index
    } 
    return  automan
    
    

