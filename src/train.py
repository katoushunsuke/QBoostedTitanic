import pandas as pd
import numpy as np
import openjij as oj
import warnings
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn import metrics
warnings.filterwarnings('ignore')

from os.path import abspath, dirname, join
import sys
root_dir = dirname(abspath(join(__file__, "../")))
sys.path.append(root_dir)

from src.preprocessing.preprocessor import Preprocessor
from src.qboost.qbooster import QBooster

def main() -> None:
    # load data ###############################################################
    train_data_path = join(root_dir, "data/train.csv")
    train_csv = pd.read_csv(train_data_path)

    # preprocessing ###########################################################
    preprocessor = Preprocessor(dataset=train_csv)
    # fill missing values
    preprocessor.fillna_by_median(column="Age")
    preprocessor.fillna_by_median(column="Fare")
    preprocessor.fillna_by_specific_value(column="Embarked", value="S")
    
    preprocessor.create_title() # create title (like Mr, Mrs, Miss, Master)
    preprocessor.grouping_familiy() # grouping family

    selected_features = ['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilyLabel']
    train_csv_post = preprocessor.get_encoded_dataset(selected_features)
    
    train_csv_post['Survived'] = (train_csv_post['Survived']-0.5) * 2 # {0, 1} -> {-1, 1}
    train_csv_post = train_csv_post[train_csv_post['Survived'].notnull()] # extract rows that have Survived values
 
    # train ###################################################################
    X = train_csv_post.values[:, 1:]
    y = train_csv_post.values[:, 0]
    y = y.astype(int)

    num_clf = 32 # number of classifiers
    models = [DTC(splitter="random",max_depth=1) for _ in range(num_clf)]

    # select features randomly and train models
    feature_idx_list = []
    for model in models:
        feature_num = np.random.randint(1, X.shape[1]) # number of features
        train_idx = np.random.choice(np.arange(X.shape[1]), feature_num) # index of features
        feature_idx_list.append(train_idx)
        model.fit(X=X[:, train_idx], y=y) 

    # test the base model
    y_pred_list_test = [] # list of predictions
    for model, feature_idx in zip(models, feature_idx_list):
        y_pred_list_test.append(model.predict(X[:, feature_idx]))
    
    y_pred_list_test = np.array(y_pred_list_test)
    y_pred_test = np.sign(np.sum(y_pred_list_test,axis=0))
    acc_test_base = metrics.accuracy_score(y_true=y, y_pred=y_pred_test)
    print("Base model accuracy: ", acc_test_base)

    # QBoost ##################################################################
    qboost = QBooster(y_train=y, ys_pred=y_pred_list_test)
    qubo = qboost.to_qubo(norm_param=3)[0]

    # solve QUBO
    sampler = oj.SASampler()
    sample_set = sampler.sample_qubo(Q=qubo, num_reads=500, num_sweeps=10000)
    raw_solution = sample_set.first.sample
    
    solution = [raw_solution[f'weight[{i}]'] for i in sorted(int(key.split('[')[1].split(']')[0]) \
        for key in raw_solution.keys())]

    # test the QBoosted model
    qboosted_y_pred_list_test = []
    for model, feature_idx, w in zip(models, feature_idx_list, solution):
        qboosted_y_pred_list_test.append(w * model.predict(X[:, feature_idx]))
    
    qboosted_y_pred_list_test = np.array(qboosted_y_pred_list_test)
    qboosted_y_pred_test = np.sign(np.sum(qboosted_y_pred_list_test,axis=0))
    qboosted_y_pred_test[qboosted_y_pred_test == 0] = -1
    acc_test_qboost = metrics.accuracy_score(y_true=y, y_pred=qboosted_y_pred_test)
    print("QBoosted model accuracy: ", acc_test_qboost)
    
    """
    # for submission ###########################################################
    test_data_path = join(root_dir, "data/test.csv")
    test_csv = pd.read_csv(test_data_path)
    
    preprocessor = Preprocessor(dataset=test_csv)
    preprocessor.fillna_by_median(column="Age")
    preprocessor.fillna_by_median(column="Fare")
    preprocessor.fillna_by_specific_value(column="Embarked", value="S")
    
    preprocessor.create_title() # create title (like Mr, Mrs, Miss, Master)
    preprocessor.grouping_familiy() # grouping family
    
    selected_features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilyLabel']
    test_csv_post = preprocessor.get_encoded_dataset(selected_features)
    
    X_test = test_csv_post.values[:, :]
    y_pred_list_test = [] # list of predictions
    for model, feature_idx, w in zip(models, feature_idx_list, solution):
        y_pred_list_test.append(w * model.predict(X_test[:, feature_idx]))
        
    y_pred_list_test = np.array(y_pred_list_test)
    y_pred_test = np.sign(np.sum(y_pred_list_test,axis=0))
    
    y_pred_test = (y_pred_test+1) / 2 # {-1, 1} -> {0, 1}
    y_pred_test = y_pred_test.astype(int)
    
    test_csv_post['PassengerId'] = test_csv['PassengerId']
    test_csv_post['Survived'] = y_pred_test
    submission_path = join(root_dir, "data/submission.csv")
    test_csv_post[['PassengerId', 'Survived']].to_csv(submission_path, index=False)
    """
    return None

if __name__ == "__main__":
    main()