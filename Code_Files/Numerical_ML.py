# %%
import pandas as pd
data= pd.read_csv("/home/nalin21478/BTP/ML-food-Processing/Numerical_Textual_ML/Data/65 Nuts/65_Nuts_Embeddings_GPT.csv",index_col=0)

# %%
data.head()

# %%
data.head()

# %%
from sklearn.model_selection import train_test_split
X=data.drop(columns=['novaclass'])
y=data['novaclass']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
path_smote="/home/nalin21478/BTP/ML-food-Processing/Numerical_Textual_ML/65 Nuts/Models/Models/GPT/Smote"
path_strat="/home/nalin21478/BTP/ML-food-Processing/Numerical_Textual_ML/65 Nuts/Models/Models/GPT/Strat"
path_smote_strat="/home/nalin21478/BTP/ML-food-Processing/Numerical_Textual_ML/65 Nuts/Models/Models/GPT/Smote_Strat"

path_smote_strat_ROC = "/home/nalin21478/BTP/ML-food-Processing/Numerical_Textual_ML/65 Nuts/Models/Models/GPT/Smote_Strat/Plots"
path_strat_ROC = "/home/nalin21478/BTP/ML-food-Processing/Numerical_Textual_ML/65 Nuts/Models/Models/GPT/Strat/Plots"
path_smote_ROC = "/home/nalin21478/BTP/ML-food-Processing/Numerical_Textual_ML/65 Nuts/Models/Models/GPT/Smote/Plots"


# %%
# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE

def tune_decision_tree_hyperparameters(X_train, y_train, X_test, y_test):
    param_dist = {
        'criterion': ['gini', 'entropy'],
        # 'splitter': ['best', 'random'],
        # 'max_depth': range(1, 101),
        # 'min_samples_split': range(2, 101),
        #'min_samples_leaf': range(1, 11),
        #'max_features': ['sqrt', 'log2'],
        # 'class_weight': ['balanced', None],
    }

    dt_classifier = DecisionTreeClassifier(random_state=42)



    y_train
    X_train, y_train = smote.fit_resample(X_train, y_train)

    cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    randomized_search = RandomizedSearchCV(
        dt_classifier, param_distributions=param_dist, cv=cv_strategy, scoring='accuracy', n_iter=50, random_state=42
    )

    randomized_search.fit(X_train, y_train)

    results_df = pd.DataFrame(randomized_search.cv_results_)

    print(f"Best Hyperparameters: {randomized_search.best_params_}")

    y_train_pred = randomized_search.best_estimator_.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Train Accuracy: {train_accuracy}")

    y_test_pred = randomized_search.best_estimator_.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy}")
    
    return randomized_search.best_params_


# Assuming X_train, y_train, X_test, and y_test are availdtle
params = tune_decision_tree_hyperparameters(X_train, y_train, X_test, y_test)
from Utility_model import evaluate_classifier_with_stratified_kfold, evaluate_classifier_with_kfold_smote, evaluate_classifier_with_stratified_smote
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(**params)
evaluate_classifier_with_stratified_smote(X_train, y_train, X_test, y_test, dt, num_folds=10,save_path=path_smote_strat,model_name="dt_classifier_smote_stratified")
evaluate_classifier_with_stratified_kfold(X_train, y_train, X_test, y_test, dt, num_folds=10,save_path=path_strat,model_name="dt_classifier_stratified")
evaluate_classifier_with_kfold_smote(X_train, y_train, X_test, y_test, dt, num_folds=10,save_path=path_smote,model_name="dt_classifier_smote")
from aup_roc import plot_roc_and_pr_curves_multiclass_smote_strat, plot_roc_and_pr_curves_multiclass_strat, plot_roc_and_pr_curves_multiclass_smote_kfold
plot_roc_and_pr_curves_multiclass_smote_strat(dt, X, y, n_splits=10, save_folder=path_smote_strat_ROC, model_name="dt_classifier_smote_stratified")
plot_roc_and_pr_curves_multiclass_strat(dt, X, y, n_splits=10, save_folder=path_strat_ROC, model_name="dt_classifier_stratified")
plot_roc_and_pr_curves_multiclass_smote_kfold(dt, X, y, n_splits=10, save_folder=path_smote_ROC, model_name="dt_classifier_smote")


# %%
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

def tune_extra_trees_hyperparameters(X_train, y_train, X_test, y_test):
    param_dist = {
        'n_estimators': np.arange(50, 1001, 50),
        'max_depth': np.arange(3, 12),
        # 'min_samples_split': np.arange(2, 22),
        # 'min_samples_leaf': np.arange(1, 22),
        # 'bootstrap': [True, False],
        # 'max_features': ['auto', 'sqrt', 'log2', None]
    }

    et_classifier = ExtraTreesClassifier(random_state=42)

    scaler = StandardScaler()
    y_train

    X_train_scaled = scaler.fit_transform(X_train)
    y_train

    X_test_scaled = scaler.transform(X_test)

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        et_classifier, param_distributions=param_dist, n_iter=50, cv=cv_strategy, scoring='accuracy', random_state=42
    )
    
    random_search.fit(X_train, y_train)

    results_df = pd.DataFrame(random_search.cv_results_)

    print(f"Best Hyperparameters: {random_search.best_params_}")
    print(f"Best Accuracy: {random_search.best_score_}")

    y_train_pred = random_search.best_estimator_.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Train Accuracy: {train_accuracy}")

    y_test_pred = random_search.best_estimator_.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy}")
    return random_search.best_params_

# Assuming X_train, y_train, X_test, and y_test are available
params=tune_extra_trees_hyperparameters(X_train, y_train, X_test, y_test)

from Utility_model import evaluate_classifier_with_stratified_kfold, evaluate_classifier_with_kfold_smote, evaluate_classifier_with_stratified_smote
from sklearn.ensemble import ExtraTreesClassifier
et=ExtraTreesClassifier(**params)


# %%
evaluate_classifier_with_stratified_smote(X_train, y_train, X_test, y_test, et, num_folds=10,save_path=path_smote_strat,model_name="et_classifier_smote_stratified")
evaluate_classifier_with_stratified_kfold(X_train, y_train, X_test, y_test, et, num_folds=10,save_path=path_strat,model_name="et_classifier_stratified")
evaluate_classifier_with_kfold_smote(X_train, y_train, X_test, y_test, et, num_folds=10,save_path=path_smote,model_name="et_classifier_smote")
from aup_roc import plot_roc_and_pr_curves_multiclass_smote_strat, plot_roc_and_pr_curves_multiclass_strat, plot_roc_and_pr_curves_multiclass_smote_kfold
plot_roc_and_pr_curves_multiclass_smote_strat(et, X, y, n_splits=10, save_folder=path_smote_strat_ROC, model_name="et_classifier_smote_stratified")
plot_roc_and_pr_curves_multiclass_strat(et, X, y, n_splits=10, save_folder=path_strat_ROC, model_name="et_classifier_stratified")
plot_roc_and_pr_curves_multiclass_smote_kfold(et, X, y, n_splits=10, save_folder=path_smote_ROC, model_name="et_classifier_smote")


# %%
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd

def tune_gradient_boosting_hyperparameters(X_train, y_train, X_test, y_test):
    param_dist = {
        'n_estimators': range(50, 1001),
        'learning_rate': (0.001, 0.3),
        'max_depth': range(3, 12),
        'min_samples_split': range(2, 22),
        'min_samples_leaf': range(1, 22),
        'subsample': (0.8, 1.0),
        'max_features': ['sqrt', 'log2']
    }

    gb_classifier = GradientBoostingClassifier(random_state=42)

    scaler = StandardScaler()
    y_train

    X_train_scaled = scaler.fit_transform(X_train)
    y_train

    X_test_scaled = scaler.transform(X_test)

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    randomized_search = RandomizedSearchCV(
        gb_classifier, param_distributions=param_dist, n_iter=50, cv=cv_strategy, scoring='accuracy', random_state=42
    )
    
    randomized_search.fit(X_train, y_train)

    results_df = pd.DataFrame(randomized_search.cv_results_)

    results_df.to_csv('hyperparameter_tuning_results_randomized_gb.csv', index=False)

    print(f"Best Hyperparameters: {randomized_search.best_params_}")
    print(f"Best Accuracy: {randomized_search.best_score_}")

    y_train_pred = randomized_search.best_estimator_.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Train Accuracy: {train_accuracy}")

    y_test_pred = randomized_search.best_estimator_.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy}")
    return randomized_search.best_params_

# Assuming X_train, y_train, X_test, and y_test are available
params = tune_gradient_boosting_hyperparameters(X_train, y_train, X_test, y_test)
from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier(**params)
from Utility_model_training import evaluate_classifier_with_stratified_kfold, evaluate_classifier_with_kfold_smote, evaluate_classifier_with_stratified_smote
evaluate_classifier_with_stratified_smote(X_train, y_train, X_test, y_test, gb, num_folds=10,save_path=path_smote_strat,model_name="gb_classifier_smote_stratified")
evaluate_classifier_with_stratified_kfold(X_train, y_train, X_test, y_test, gb, num_folds=10,save_path=path_strat,model_name="gb_classifier_stratified")
evaluate_classifier_with_kfold_smote(X_train, y_train, X_test, y_test, gb, num_folds=10,save_path=path_smote,model_name="gb_classifier_smote")
from aup_roc import plot_roc_and_pr_curves_multiclass_smote_strat, plot_roc_and_pr_curves_multiclass_strat, plot_roc_and_pr_curves_multiclass_smote_kfold
plot_roc_and_pr_curves_multiclass_smote_strat(gb, X, y, n_splits=10, save_folder=path_smote_strat_ROC, model_name="gb_classifier_smote_stratified")
plot_roc_and_pr_curves_multiclass_strat(gb, X, y, n_splits=10, save_folder=path_strat_ROC, model_name="gb_classifier_stratified")
plot_roc_and_pr_curves_multiclass_smote_kfold(gb, X, y, n_splits=10, save_folder=path_smote_ROC, model_name="gb_classifier_smote")


# %%
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
# use random search instead of grid search for hyperparameter optimization
from sklearn.model_selection import RandomizedSearchCV

def tune_LGBM_hyperparameters(X_train, y_train, X_test, y_test):
    param_dist = {
        'n_estimators': (10, 10000,100),

    }

    lgbm_classifier = LGBMClassifier(random_state=42)

    scaler = StandardScaler()
    y_train

    X_train_scaled = scaler.fit_transform(X_train)
    y_train

    X_test_scaled = scaler.transform(X_test)

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    bayes_search = RandomizedSearchCV(
        lgbm_classifier, param_dist, n_iter=50, cv=cv_strategy, scoring='accuracy', random_state=42
    )
    
    bayes_search.fit(X_train, y_train)

    print(f"Best Hyperparameters: {bayes_search.best_params_}")
    print(f"Best Accuracy: {bayes_search.best_score_}")

    y_train_pred = bayes_search.best_estimator_.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Train Accuracy: {train_accuracy}")

    y_test_pred = bayes_search.best_estimator_.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy}")
    return bayes_search.best_params_

params=tune_LGBM_hyperparameters(X_train, y_train, X_test, y_test)
from lightgbm import LGBMClassifier
lgbm=LGBMClassifier(**params)
from Utility_model_training import evaluate_classifier_with_stratified_kfold, evaluate_classifier_with_kfold_smote, evaluate_classifier_with_stratified_smote
evaluate_classifier_with_stratified_smote(X_train, y_train, X_test, y_test, lgbm, num_folds=10,save_path=path_smote_strat,model_name="lgbm_classifier_smote_stratified")
evaluate_classifier_with_stratified_kfold(X_train, y_train, X_test, y_test, lgbm, num_folds=10,save_path=path_strat,model_name="lgbm_classifier_stratified")
evaluate_classifier_with_kfold_smote(X_train, y_train, X_test, y_test, lgbm, num_folds=10,save_path=path_smote,model_name="lgbm_classifier_smote")
from aup_roc import plot_roc_and_pr_curves_multiclass_smote_strat, plot_roc_and_pr_curves_multiclass_strat, plot_roc_and_pr_curves_multiclass_smote_kfold
plot_roc_and_pr_curves_multiclass_smote_strat(lgbm, X, y, n_splits=10, save_folder=path_smote_strat_ROC, model_name="lgbm_classifier_smote_stratified")
plot_roc_and_pr_curves_multiclass_strat(lgbm, X, y, n_splits=10, save_folder=path_strat_ROC, model_name="lgbm_classifier_stratified")
plot_roc_and_pr_curves_multiclass_smote_kfold(lgbm, X, y, n_splits=10, save_folder=path_smote_ROC, model_name="lgbm_classifier_smote")

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
import pandas as pd

def tune_RF_hyperparameters(X_train, y_train, X_test, y_test):
    param_dist = {
        'n_estimators': range(10, 501),
        'criterion': ['gini', 'entropy'],
        'max_depth': range(1, 21),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }

    rf_classifier = RandomForestClassifier(random_state=42)

    scaler = StandardScaler()
    y_train

    X_train_scaled = scaler.fit_transform(X_train)
    y_train

    X_test_scaled = scaler.transform(X_test)

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        rf_classifier, param_distributions=param_dist, n_iter=50, cv=cv_strategy, scoring='accuracy', random_state=42
    )
    
    random_search.fit(X_train, y_train)

    results_df = pd.DataFrame(random_search.cv_results_)

    print(f"Best Hyperparameters: {random_search.best_params_}")
    print(f"Best Accuracy: {random_search.best_score_}")

    y_train_pred = random_search.best_estimator_.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Train Accuracy: {train_accuracy}")

    y_test_pred = random_search.best_estimator_.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy}")
    return random_search.best_params_



params=tune_RF_hyperparameters(X_train, y_train, X_test, y_test)
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(**params)
from Utility_model_training import evaluate_classifier_with_stratified_kfold, evaluate_classifier_with_kfold_smote, evaluate_classifier_with_stratified_smote
evaluate_classifier_with_stratified_smote(X_train, y_train, X_test, y_test, rf, num_folds=10,save_path=path_smote_strat,model_name="rf_classifier_smote_stratified")
evaluate_classifier_with_stratified_kfold(X_train, y_train, X_test, y_test, rf, num_folds=10,save_path=path_strat,model_name="rf_classifier_stratified")
evaluate_classifier_with_kfold_smote(X_train, y_train, X_test, y_test, rf, num_folds=10,save_path=path_smote,model_name="rf_classifier_smote")
from aup_roc import plot_roc_and_pr_curves_multiclass_smote_strat, plot_roc_and_pr_curves_multiclass_strat, plot_roc_and_pr_curves_multiclass_smote_kfold
plot_roc_and_pr_curves_multiclass_smote_strat(rf, X, y, n_splits=10, save_folder=path_smote_strat_ROC, model_name="rf_classifier_smote_stratified")
plot_roc_and_pr_curves_multiclass_strat(rf, X, y, n_splits=10, save_folder=path_strat_ROC, model_name="rf_classifier_stratified")
plot_roc_and_pr_curves_multiclass_smote_kfold(rf, X, y, n_splits=10, save_folder=path_smote_ROC, model_name="rf_classifier_smote")

# %%
y_train_xg=y_train-1
y_test_xg=y_test-1
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd

def tune_xgboost_hyperparameters(X, y):
    param_dist = {
        'max_depth': range(1, 20,2),
        'learning_rate': [0.01, 0.05, 0.1,0.001,0.0001],
        'n_estimators': range(50, 500,10),
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 1, 2, 3, 4],
        'reg_alpha': [0, 0.1, 0.5, 1, 2],
        'reg_lambda': [0, 0.1, 0.5, 1, 2]
    }

    xgb_classifier = XGBClassifier()
    random_search = RandomizedSearchCV(xgb_classifier, param_distributions=param_dist, cv=5, scoring='accuracy')
    random_search.fit(X, y)


    print(f"Best Hyperparameters: {random_search.best_params_}")
    print(f"Best Accuracy: {random_search.best_score_}")

    return random_search.best_params_

params=tune_xgboost_hyperparameters(X_train, y_train_xg)

#writw thw params to a .txt
with open("xgboost_hyperparameters.txt", "w") as file:
    file.write(str(params))
    
from xgboost import XGBClassifier

xgb=XGBClassifier(**params)
from Utility_model_training import evaluate_classifier_with_stratified_kfold, evaluate_classifier_with_kfold_smote, evaluate_classifier_with_stratified_smote
evaluate_classifier_with_stratified_smote(X_train, y_train, X_test, y_test, xgb, num_folds=10,save_path=path_smote_strat,model_name="xgb_classifier_smote_stratified")
evaluate_classifier_with_stratified_kfold(X_train, y_train, X_test, y_test, xgb, num_folds=10,save_path=path_strat,model_name="xgb_classifier_stratified")
evaluate_classifier_with_kfold_smote(X_train, y_train, X_test, y_test, xgb, num_folds=10,save_path=path_smote,model_name="xgb_classifier_smote")
y=y-1
from aup_roc import plot_roc_and_pr_curves_multiclass_smote_strat, plot_roc_and_pr_curves_multiclass_strat, plot_roc_and_pr_curves_multiclass_smote_kfold
plot_roc_and_pr_curves_multiclass_smote_strat(xgb, X, y, n_splits=10, save_folder=path_smote_strat_ROC, model_name="xg_KFold_SMOTE_strat")
plot_roc_and_pr_curves_multiclass_strat(xgb, X, y, n_splits=10, save_folder=path_strat_ROC, model_name="xg_KFold_Strat")
plot_roc_and_pr_curves_multiclass_smote_kfold(xgb, X, y, n_splits=10, save_folder=path_smote_ROC, model_name="xg_KFold_SMOTE")



