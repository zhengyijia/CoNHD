import numpy as np

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix

def get_clf_eval(y_test, pred, avg='micro', outputdim=None):
    if outputdim is not None:
        confusion = confusion_matrix(y_test, pred, labels=np.arange(outputdim))
    else:
        confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average=avg)
    recall = recall_score(y_test, pred, average=avg)
    f1 = f1_score(y_test, pred, average=avg)
    print('Confusion Matrix')
    print(confusion)
    print('Accuracy:{}, Precision:{}, Recall:{}, F1:{}'.format(accuracy, precision, recall, f1))
    return confusion, accuracy, precision, recall, f1

