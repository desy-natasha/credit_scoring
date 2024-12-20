from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt

def model(models,X_train,X_test,y_train,y_test,pprint=True):
    models.fit(X_train,y_train)
    
    list_features = X_train.columns
    list_score = []
    
    y_pred = models.predict(X_test)   

    # evaluation metrics
    accuracy =  round(accuracy_score(y_test,y_pred),3)
    precision = round(precision_score(y_test,y_pred),3)
    recall = round(recall_score(y_test,y_pred),3)
    f1 = round(f1_score(y_test, y_pred),3)
    auc = round(roc_auc_score(y_test, y_pred),3)
    confusion_mat = confusion_matrix(y_test, y_pred)

    if pprint:
        print(f'Accuracy: {accuracy} | Precision: {precision} | Recall: {recall} | F1-Score: {f1} | AUC: {auc} ')
        print(f'Confusion Matrix: \n {confusion_mat}')
    
    # predict the test sample result and compare with the actual value
    y_pred_proba = models.predict_proba(X_test)[:,1] 
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize = [5,5])
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    return models,accuracy,precision,recall,f1,auc

def random_search(parameters,X,y,model,iter=100,cv_default = True):
    scaler = StandardScaler()
    scaler.fit(X)
    
    if cv_default:
        cv = 5
    else:
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)
    
    # scoring metric for evaluation
    score = {'f1_score': 'f1'}
    random_cv = RandomizedSearchCV(model, param_distributions=parameters, scoring=score, cv=cv, verbose=False,n_iter=iter, refit='f1_score')

    random_cv.fit(scaler.transform(X), y)
    print(f'Best Parameters: {random_cv.best_estimator_}')
    return (random_cv.best_estimator_)
