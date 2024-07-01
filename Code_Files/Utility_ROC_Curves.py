import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_roc_curves_multiclass_smote_strat(model, X, y, n_splits=5, save_folder=None, model_name=None):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    y_bin = label_binarize(y, classes=np.unique(y))
    n_classes = y_bin.shape[1]

    scaler = StandardScaler()
    
    # Define a custom color palette
    custom_palette = {1: 'green', 2: '#ff9100', 3: 'purple', 4: 'brown'}
  # Add more colors if needed

    # Set font properties
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 12

    plt.figure(figsize=(8, 8))

    for class_index in range(n_classes):
        mean_fpr = np.linspace(0, 1, 100)
        tpr_list = []
        aucs = []  
        
        for i, (train, test) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train].values, X.iloc[test].values
            y_train, y_test = y.iloc[train], y.iloc[test]

            # Apply scaling
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Apply SMOTE
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

            model.fit(X_train_resampled, y_train_resampled)

            y_prob = model.predict_proba(X_test_scaled)

            fpr, tpr, _ = roc_curve(y_bin[test][:, class_index], y_prob[:, class_index])
            tpr_list.append(np.interp(mean_fpr, fpr, tpr))
            tpr_list[-1][0] = 0.0 
            
            auc_score = auc(fpr, tpr)
            aucs.append(auc_score) 

        mean_tpr = np.mean(tpr_list, axis=0)
        mean_tpr[-1] = 1.0  

        mean_auc = np.mean(aucs)

        # Use custom color for each class
        color = custom_palette[class_index + 1]

        plt.plot(mean_fpr, mean_tpr, label=f'Class {class_index + 1} (AUC = {mean_auc:.2f})', color=color)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(os.path.join(save_folder, 'roc_curves_'+model_name+'.png'))
    else:
        plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler
import os

def plot_roc_curves_multiclass_strat(model, X, y, n_splits=5, save_folder=None, model_name=None):
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
        mean_fpr = np.linspace(0, 1, 100)
        tpr_list = []
        aucs = []  
        
        for i, (train, test) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train].values, X.iloc[test].values
            y_train, y_test = y.iloc[train], y.iloc[test]

            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model.fit(X_train_scaled, y_train)

            y_prob = model.predict_proba(X_test_scaled)

            fpr, tpr, _ = roc_curve(y_bin[test][:, class_index], y_prob[:, class_index])
            tpr_list.append(np.interp(mean_fpr, fpr, tpr))
            tpr_list[-1][0] = 0.0 
            
            auc_score = auc(fpr, tpr)
            aucs.append(auc_score)

        mean_tpr = np.mean(tpr_list, axis=0)
        mean_tpr[-1] = 1.0  

        mean_auc = np.mean(aucs)

        # Use custom color for each class
        color = custom_palette[class_index + 1]

        plt.plot(mean_fpr, mean_tpr, label=f'Class {class_index+1} (AUC = {mean_auc:.2f})', color=color)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(os.path.join(save_folder, 'roc_curves_'+model_name+'.png'))
    else:
        plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os

def plot_roc_curves_multiclass_smote(model, X, y, n_splits=5, save_folder=None, model_name=None):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    y_bin = label_binarize(y, classes=np.unique(y))
    n_classes = y_bin.shape[1]

    scaler = StandardScaler()

    # Set font properties
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 12

    # Define a custom color palette
    custom_palette = {1: 'green', 2: '#ff9100', 3: 'purple', 4: 'brown'} # Add more colors if needed

    plt.figure(figsize=(8, 8))

    for class_index in range(n_classes):
        mean_fpr = np.linspace(0, 1, 100)
        tpr_list = []
        aucs = []  
        
        for i, (train, test) in enumerate(kf.split(X, y)):
            X_train, X_test = X.iloc[train].values, X.iloc[test].values
            y_train, y_test = y.iloc[train], y.iloc[test]

            # Apply scaling
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Apply SMOTE
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

            model.fit(X_train_resampled, y_train_resampled)

            y_prob = model.predict_proba(X_test_scaled)

            fpr, tpr, _ = roc_curve(y_bin[test][:, class_index], y_prob[:, class_index])
            tpr_list.append(np.interp(mean_fpr, fpr, tpr))
            tpr_list[-1][0] = 0.0 
            
            auc_score = auc(fpr, tpr)
            aucs.append(auc_score)

        mean_tpr = np.mean(tpr_list, axis=0)
        mean_tpr[-1] = 1.0  

        mean_auc = np.mean(aucs)

        # Use custom color for each class
        color = custom_palette[class_index + 1]

        plt.plot(mean_fpr, mean_tpr, label=f'Class {class_index+1} (AUC = {mean_auc:.2f})', color=color)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(os.path.join(save_folder, 'roc_curves_'+model_name+'.png'))
    else:
        plt.show()



