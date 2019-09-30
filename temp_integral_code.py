import warnings
warnings.filterwarnings('ignore') #忽略warning
import pandas as pd
import numpy as np
from sklearn import ensemble, tree, neighbors, linear_model, svm, naive_bayes
from sklearn import model_selection
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
import sys
from sklearn import preprocessing
import os

#检测输入数据集中是否存在空值
def checkNanData(file):
    if file.isnull().any() == True:
        raise ValueError('Null values exist in this file,文件存在空值')

#标准化数据，使用sklearn.preprocessing.StandardScaler方法
def processingData(file):
    scaler = preprocessing.StandardScaler()
    file = scaler.fit_transform(file)

#画ROC曲线
def tune_roc_curve(classifier):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr,
                 tpr,
                 lw=1,
                 alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1],
             linestyle='--',
             lw=2,
             color='r',
             label='Chance',
             alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr,
             mean_tpr,
             color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2,
             alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr,
                     tprs_lower,
                     tprs_upper,
                     color='grey',
                     alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Tune Hyperparameter {} ROC'.format(
        classifier.__class__.__name__))
    plt.legend(loc="lower right")
	#高清输出
    plt.tight_layout()
	#保存
    plt.savefig(os.path.join(
        path, 'Tune_hp {}.pdf'.format(classifier.__class__.__name__)),
                format='pdf')
    plt.close()

#预测结果输出，包括预测值，预测概率值
def pred_ft_gene(clf):
    pred = clf.predict(testData.values)
    pred_pro = clf.predict_proba(testData.values)
    predictions = pd.DataFrame(
        {
            'label': pred,
            'ft probability': pred_pro[:, 1],
            'no-ft probility': pred_pro[:, 0]
        },
        index=testData.index)
    predictions.sort_values(by=['ft probability'],
                            ascending=False,
                            inplace=True)
    predictions.to_excel(
        os.path.join(path,
                     '{} prediction.xlsx'.format(clf.__class__.__name__)))

#获取当先路径
path = os.getcwd()
#sys.argv[1]为train数据集，sys.argv[2]为预测数据集
DataSet = pd.read_csv(sys.argv[1], sep='\t', index_col='gene')
testData = pd.read_csv(sys.argv[2], sep='\t', index_col='gene')
#划分样本集和target，并转化为ndarray格式
X = DataSet.drop(['label'], axis=1).values
y = DataSet.label.values
#调用processingData函数
processingData(X)
processingData(testData)
#交叉验证方式采用StratifiedKFold，分层采样，交叉划分
cv = StratifiedKFold(n_splits=5, random_state=42)

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1_score': make_scorer(f1_score),
    'auc': make_scorer(roc_auc_score)
}
#各个算法的在网格调参的参数
lr_param = {
    'fit_intercept': [True, False],  #default: True
    #'penalty': ['l1','l2'],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag',
               'saga'],  #default: lbfgs
    'random_state': [0]
}
knn_param = {
    'n_neighbors': [1, 2, 3, 4, 5, 6, 7],  #default: 5
    'weights': ['uniform', 'distance'],  #default = ‘uniform’
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}
svm_param = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [1, 2, 3, 4, 5],  #default=1.0
    'gamma': [.1, .25, .5, .75, 1.0],  #edfault: auto
    'decision_function_shape': ['ovo', 'ovr'],  #default:ovr
    'probability': [True],
    'random_state': [0]
}
adaboost_param = {
    'n_estimators': [10, 50, 100, 300],  #default=50
    'learning_rate': [.01, .03, .05, .1, .25],  #default=1
    #'algorithm': ['SAMME', 'SAMME.R'], #default=’SAMME.R
    'random_state': [0]
}
bagging_param = {
    'n_estimators': [10, 50, 100, 300],  #default=10
    'max_samples': [.1, .25, .5, .75, 1.0],  #default=1.0
    'random_state': [0]
}
xgb_param = {
    'learning_rate': [.01, .03, .05, .1, .25],  #default: .3
    'max_depth': [1, 2, 4, 6, 8, 10],  #default 2
    'n_estimators': [10, 50, 100, 300],
    'seed': [0]
}

lr = linear_model.LogisticRegressionCV()
knn = neighbors.KNeighborsClassifier()
adaboost = ensemble.AdaBoostClassifier()
bagging = ensemble.BaggingClassifier()
xgb = XGBClassifier()
#网格调参，评分采用AUC
lr_gs = GridSearchCV(estimator=lr,
                     param_grid=lr_param,
                     cv=cv,
                     scoring='roc_auc',
                     refit=True)
knn_gs = GridSearchCV(estimator=knn,
                      param_grid=knn_param,
                      cv=cv,
                      scoring='roc_auc',
                      refit=True)
svm_gs = GridSearchCV(estimator=svm.SVC(),
                      param_grid=svm_param,
                      cv=cv,
                      scoring='roc_auc',
                      refit=True)
ada_gs = GridSearchCV(estimator=adaboost,
                      param_grid=adaboost_param,
                      cv=cv,
                      scoring='roc_auc',
                      refit=True)
bagging_gs = GridSearchCV(estimator=bagging,
                          param_grid=bagging_param,
                          cv=cv,
                          scoring='roc_auc',
                          refit=True)
xgb_gs = GridSearchCV(estimator=xgb,
                      param_grid=xgb_param,
                      cv=cv,
                      scoring='roc_auc',
                      refit=True)

lr_gs = lr_gs.fit(X, y)
knn_gs = knn_gs.fit(X, y)
svm_gs = svm_gs.fit(X, y)
ada_gs = ada_gs.fit(X, y)
bagging_gs = bagging_gs.fit(X, y)
xgb_gs = xgb_gs.fit(X, y)
#获取最佳模型
Blr = lr_gs.best_estimator_
Bknn = knn_gs.best_estimator_
Bsvm = svm_gs.best_estimator_
Bada = ada_gs.best_estimator_
Bbagging = bagging_gs.best_estimator_
Bxgb = xgb_gs.best_estimator_
best_model = [Blr, Bknn, Bsvm, Bada, Bbagging, Bxgb]

for model in best_model:
    tune_roc_curve(model)

for model in best_model:
    pred_ft_gene(model)