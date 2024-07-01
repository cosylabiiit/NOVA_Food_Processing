import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import precision_recall_curve, auc
from imblearn.over_sampling import SMOTE

from sklearn.metrics import auc, precision_recall_curve

def plot_pr_curves_multiclass_strat(model, X, y, n_splits=5, save_folder=None, model_name=None):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    y_bin = label_binarize(y, classes=np.unique(y))
    n_classes = y_bin.shape[1]

    scaler = StandardScaler()

    # Set font properties
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 12

    # Define a custom color palette
    custom_palette = {1: 'green', 2: '#ff9100', 3: 'purple', 4: 'brown'}  # Add more colors if needed

    plt.figure(figsize=(8, 8))

    for class_index in range(n_classes):
        mean_recall = np.linspace(0, 1, 100)
        precision_list = []
        aups = []  
        
        for i, (train, test) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train].values, X.iloc[test].values
            y_train, y_test = y.iloc[train], y.iloc[test]

            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model.fit(X_train_scaled, y_train)

            y_prob = model.predict_proba(X_test_scaled)

            precision, recall, _ = precision_recall_curve(y_bin[test][:, class_index], y_prob[:, class_index])
            precision_list.append(np.interp(mean_recall, recall[::-1], precision[::-1]))
            precision_list[-1][0] = 1.0 
            
            aup_score = auc(recall, precision)
            aups.append(aup_score)

        mean_precision = np.mean(precision_list, axis=0)
        mean_precision[-1] = 0.0  

        mean_aup = np.mean(aups)

        # Use custom color for each class
        color = custom_palette[class_index + 1]

        plt.plot(mean_recall, mean_precision, label=f'Class {class_index+1} (AUP = {mean_aup:.2f})', color=color)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower right')
    
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(os.path.join(save_folder, 'pr_curves_'+model_name+'.png'))
    else:
        plt.show()
        
        
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc, precision_recall_curve
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize, StandardScaler

def plot_pr_curves_multiclass_strat_smote(model, X, y, n_splits=5, save_folder=None, model_name=None):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    y_bin = label_binarize(y, classes=np.unique(y))
    n_classes = y_bin.shape[1]

    scaler = StandardScaler()
    smote = SMOTE(random_state=42)

    # Set font properties
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 12

    # Define a custom color palette
    custom_palette = {1: 'green', 2: '#ff9100', 3: 'purple', 4: 'brown'}  # Add more colors if needed

    plt.figure(figsize=(8, 8))

    for class_index in range(n_classes):
        mean_recall = np.linspace(0, 1, 100)
        precision_list = []
        aups = []  
        
        for i, (train, test) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train].values, X.iloc[test].values
            y_train, y_test = y.iloc[train], y.iloc[test]

            X_train_scaled, y_train_resampled = smote.fit_resample(X_train, y_train)

            X_train_scaled = scaler.fit_transform(X_train_scaled)
            X_test_scaled = scaler.transform(X_test)

            model.fit(X_train_scaled, y_train_resampled)

            y_prob = model.predict_proba(X_test_scaled)

            precision, recall, _ = precision_recall_curve(y_bin[test][:, class_index], y_prob[:, class_index])
            precision_list.append(np.interp(mean_recall, recall[::-1], precision[::-1]))
            precision_list[-1][0] = 1.0 
            
            aup_score = auc(recall, precision)
            aups.append(aup_score)

        mean_precision = np.mean(precision_list, axis=0)
        mean_precision[-1] = 0.0  

        mean_aup = np.mean(aups)

        # Use custom color for each class
        color = custom_palette[class_index + 1]

        plt.plot(mean_recall, mean_precision, label=f'Class {class_index+1} (AUP = {mean_aup:.2f})', color=color)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower right')
    
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(os.path.join(save_folder, 'pr_curves_'+model_name+'.png'))
    else:
        plt.show()
        
        

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.metrics import auc, precision_recall_curve
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize, StandardScaler

def plot_pr_curves_multiclass_kfold_smote(model, X, y, n_splits=5, save_folder=None, model_name=None):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    y_bin = label_binarize(y, classes=np.unique(y))
    n_classes = y_bin.shape[1]

    scaler = StandardScaler()
    smote = SMOTE(random_state=42)

    # Set font properties
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 12

    # Define a custom color palette
    custom_palette = {1: 'green', 2: '#ff9100', 3: 'purple', 4: 'brown'}  # Add more colors if needed

    plt.figure(figsize=(8, 8))

    for class_index in range(n_classes):
        mean_recall = np.linspace(0, 1, 100)
        precision_list = []
        aups = []  
        
        for i, (train, test) in enumerate(kf.split(X, y)):
            X_train, X_test = X.iloc[train].values, X.iloc[test].values
            y_train, y_test = y.iloc[train], y.iloc[test]

            X_train_scaled, y_train_resampled = smote.fit_resample(X_train, y_train)

            X_train_scaled = scaler.fit_transform(X_train_scaled)
            X_test_scaled = scaler.transform(X_test)

            model.fit(X_train_scaled, y_train_resampled)

            y_prob = model.predict_proba(X_test_scaled)

            precision, recall, _ = precision_recall_curve(y_bin[test][:, class_index], y_prob[:, class_index])
            precision_list.append(np.interp(mean_recall, recall[::-1], precision[::-1]))
            precision_list[-1][0] = 1.0 
            
            aup_score = auc(recall, precision)
            aups.append(aup_score)

        mean_precision = np.mean(precision_list, axis=0)
        mean_precision[-1] = 0.0  

        mean_aup = np.mean(aups)

        # Use custom color for each class
        color = custom_palette[class_index + 1]

        plt.plot(mean_recall, mean_precision, label=f'Class {class_index+1} (AUP = {mean_aup:.2f})', color=color)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower right')
    
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(os.path.join(save_folder, 'pr_curves_'+model_name+'.png'))
    else:
        plt.show()