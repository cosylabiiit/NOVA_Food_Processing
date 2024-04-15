from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, precision_recall_fscore_support,classification_report
from imblearn.over_sampling import SMOTE
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, matthews_corrcoef, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt

# function for stratified k-fold and smote
def evaluate_classifier_with_stratified_smote(X_train, y_train, X_test, y_test, classifier, num_folds=10, save_path=None, model_name=None):
    # Stratified K-Fold
    k_fold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=0)
    # lists to store the values
    accuracies_val = []
    accuracies_train = []
    train_scores = []
    val_scores = []
    train_mcc = []
    val_mcc = []
    train_f1 = []
    val_f1 = []
    train_precision = []
    val_precision = []
    train_recall = []
    val_recall = []
    # scaling the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # running the stratified kfold
    for fold_idx, (train_indices, val_indices) in enumerate(k_fold.split(X_train_scaled, y_train), 1):
        
        X_fold_train, X_fold_val = X_train_scaled[train_indices], X_train_scaled[val_indices]
        y_fold_train, y_fold_val = y_train.iloc[train_indices], y_train.iloc[val_indices]
        # using SMOTE
        smote = SMOTE(random_state=42)
        X_fold_train_resampled, y_fold_train_resampled = smote.fit_resample(X_fold_train, y_fold_train)
        # Train and val scores
        classifier.fit(X_fold_train_resampled, y_fold_train_resampled)
        train_score = accuracy_score(y_fold_train_resampled, classifier.predict(X_fold_train_resampled))
        train_scores.append(train_score)
        y_train_pred = classifier.predict(X_fold_train_resampled)
        fold_accuracy_train = accuracy_score(y_fold_train_resampled, np.round(y_train_pred))
        accuracies_train.append(fold_accuracy_train)

        train_mcc.append(matthews_corrcoef(y_fold_train_resampled, np.round(y_train_pred)))
        train_f1.append(precision_recall_fscore_support(y_fold_train_resampled, np.round(y_train_pred), average='weighted')[2])
        precision, recall, _, _ = precision_recall_fscore_support(y_fold_train_resampled, np.round(y_train_pred), average='weighted')
        train_precision.append(precision)
        train_recall.append(recall)

        val_score = accuracy_score(y_fold_val, classifier.predict(X_fold_val))
        val_scores.append(val_score)

        y_val_pred = classifier.predict(X_fold_val)
        fold_accuracy_val = accuracy_score(y_fold_val, np.round(y_val_pred))
        accuracies_val.append(fold_accuracy_val)

        
        val_mcc.append(matthews_corrcoef(y_fold_val, np.round(y_val_pred)))
        val_f1.append(precision_recall_fscore_support(y_fold_val, np.round(y_val_pred), average='weighted')[2])
        precision, recall, _, _ = precision_recall_fscore_support(y_fold_val, np.round(y_val_pred), average='weighted')
        val_precision.append(precision)
        val_recall.append(recall)
    # averaging the metrics
    average_accuracy_val = sum(accuracies_val) / num_folds
    print(f'Average Accuracy Val K-Fold: {average_accuracy_val * 100:.4f}')

    average_accuracy_train = sum(accuracies_train) / num_folds
    print(f'Average Accuracy Train K-Fold: {average_accuracy_train * 100:.4f}')

    average_mcc_val = sum(val_mcc) / num_folds
    print(f'Average MCC Val K-Fold: {average_mcc_val:.4f}')

    average_mcc_train = sum(train_mcc) / num_folds
    print(f'Average MCC Train K-Fold: {average_mcc_train:.4f}')

    average_f1_val = sum(val_f1) / num_folds
    print(f'Average F1 Score Val K-Fold: {average_f1_val:.4f}')

    average_f1_train = sum(train_f1) / num_folds
    print(f'Average F1 Score Train K-Fold: {average_f1_train:.4f}')

    average_precision_val = sum(val_precision) / num_folds
    print(f'Average Precision Val K-Fold: {average_precision_val:.4f}')

    average_precision_train = sum(train_precision) / num_folds
    print(f'Average Precision Train K-Fold: {average_precision_train:.4f}')

    average_recall_val = sum(val_recall) / num_folds
    print(f'Average Recall Val K-Fold: {average_recall_val:.4f}')

    average_recall_train = sum(train_recall) / num_folds
    print(f'Average Recall Train K-Fold: {average_recall_train:.4f}')

    y_pred = classifier.predict(X_test_scaled)
    # metrics for the test set
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy Test: {accuracy * 100:.2f}%')

    test_mcc = matthews_corrcoef(y_test, np.round(y_pred))
    print(f'MCC Test: {test_mcc:.4f}')

    test_f1 = precision_recall_fscore_support(y_test, np.round(y_pred), average='weighted')[2]
    print(f'F1 Score Test: {test_f1:.4f}')

    test_precision, test_recall, _, _ = precision_recall_fscore_support(y_test, np.round(y_pred), average='weighted')
    print(f'Precision Test: {test_precision:.4f}')
    print(f'Recall Test: {test_recall:.4f}')

    class_report = classification_report(y_test, y_pred)
    print('Classification Report:')
    print(class_report)

    conf_matrix_test = confusion_matrix(y_test, np.round(y_pred))
    print('Confusion Matrix Test:')
    print(conf_matrix_test)
    # saving the model and the outputs to a text file
    if save_path and model_name:
        output_filename = os.path.join(save_path, f"{model_name}_outputs.txt")
        with open(output_filename, 'w') as f:
            f.write(f'Average Accuracy Val K-Fold: {average_accuracy_val * 100:.4f}\n')
            f.write(f'Average Accuracy Train K-Fold: {average_accuracy_train * 100:.4f}\n')
            f.write(f'Average MCC Val K-Fold: {average_mcc_val:.4f}\n')
            f.write(f'Average MCC Train K-Fold: {average_mcc_train:.4f}\n')
            f.write(f'Average F1 Score Val K-Fold: {average_f1_val:.4f}\n')
            f.write(f'Average F1 Score Train K-Fold: {average_f1_train:.4f}\n')
            f.write(f'Average Precision Val K-Fold: {average_precision_val:.4f}\n')
            f.write(f'Average Precision Train K-Fold: {average_precision_train:.4f}\n')
            f.write(f'Average Recall Val K-Fold: {average_recall_val:.4f}\n')
            f.write(f'Average Recall Train K-Fold: {average_recall_train:.4f}\n')
            f.write(f'Accuracy Test: {accuracy * 100:.2f}%\n')
            f.write(f'MCC Test: {test_mcc:.4f}\n')
            f.write(f'F1 Score Test: {test_f1:.4f}\n')
            f.write(f'Precision Test: {test_precision:.4f}\n')
            f.write(f'Recall Test: {test_recall:.4f}\n')
            f.write('Classification Report:\n')
            f.write(class_report + '\n')
            f.write('Confusion Matrix Test:\n')
            f.write(str(conf_matrix_test) + '\n')
        print(f"Outputs saved to: {output_filename}")
        # save the model using joblib
        joblib.dump(classifier, os.path.join(save_path, f"{model_name}.joblib"))
        joblib.dump(scaler, os.path.join(save_path, f"{model_name}_scaler.joblib"))
        
        
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, matthews_corrcoef, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, matthews_corrcoef, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
# function for  k-fold and smote
def evaluate_classifier_with_kfold_smote(X_train, y_train, X_test, y_test, classifier, num_folds=10, save_path=None, model_name=None):
    # Stratified K-Fold
    k_fold = KFold(n_splits=num_folds, shuffle=True, random_state=0)
    # lists to store the values
    accuracies_val = []
    accuracies_train = []
    train_scores = []
    val_scores = []
    train_mcc = []
    val_mcc = []
    train_f1 = []
    val_f1 = []
    train_precision = []
    val_precision = []
    train_recall = []
    val_recall = []
    # scaling the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # running kfold
    for fold_idx, (train_indices, val_indices) in enumerate(k_fold.split(X_train_scaled, y_train), 1):
        X_fold_train, X_fold_val = X_train_scaled[train_indices], X_train_scaled[val_indices]
        y_fold_train, y_fold_val = y_train.iloc[train_indices], y_train.iloc[val_indices]
        # resampling
        smote = SMOTE()
        X_fold_train_resampled, y_fold_train_resampled = smote.fit_resample(X_fold_train, y_fold_train)
        # training and calculating training and validation metrics
        classifier.fit(X_fold_train_resampled, y_fold_train_resampled)
        train_score = accuracy_score(y_fold_train_resampled, classifier.predict(X_fold_train_resampled))
        train_scores.append(train_score)
        y_train_pred = classifier.predict(X_fold_train_resampled)
        fold_accuracy_train = accuracy_score(y_fold_train_resampled, np.round(y_train_pred))
        accuracies_train.append(fold_accuracy_train)

        
        train_mcc.append(matthews_corrcoef(y_fold_train_resampled, np.round(y_train_pred)))
        train_f1.append(precision_recall_fscore_support(y_fold_train_resampled, np.round(y_train_pred), average='weighted')[2])
        precision, recall, _, _ = precision_recall_fscore_support(y_fold_train_resampled, np.round(y_train_pred), average='weighted')
        train_precision.append(precision)
        train_recall.append(recall)

        val_score = accuracy_score(y_fold_val, classifier.predict(X_fold_val))
        val_scores.append(val_score)

        y_val_pred = classifier.predict(X_fold_val)
        fold_accuracy_val = accuracy_score(y_fold_val, np.round(y_val_pred))
        accuracies_val.append(fold_accuracy_val)

        
        val_mcc.append(matthews_corrcoef(y_fold_val, np.round(y_val_pred)))
        val_f1.append(precision_recall_fscore_support(y_fold_val, np.round(y_val_pred), average='weighted')[2])
        precision, recall, _, _ = precision_recall_fscore_support(y_fold_val, np.round(y_val_pred), average='weighted')
        val_precision.append(precision)
        val_recall.append(recall)

    # averaging the metrics
    average_accuracy_val = sum(accuracies_val) / num_folds
    print(f'Average Accuracy Val K-Fold: {average_accuracy_val * 100:.4f}')

    average_accuracy_train = sum(accuracies_train) / num_folds
    print(f'Average Accuracy Train K-Fold: {average_accuracy_train * 100:.4f}')

    average_mcc_val = sum(val_mcc) / num_folds
    print(f'Average MCC Val K-Fold: {average_mcc_val:.4f}')

    average_mcc_train = sum(train_mcc) / num_folds
    print(f'Average MCC Train K-Fold: {average_mcc_train:.4f}')

    average_f1_val = sum(val_f1) / num_folds
    print(f'Average F1 Score Val K-Fold: {average_f1_val:.4f}')

    average_f1_train = sum(train_f1) / num_folds
    print(f'Average F1 Score Train K-Fold: {average_f1_train:.4f}')

    average_precision_val = sum(val_precision) / num_folds
    print(f'Average Precision Val K-Fold: {average_precision_val:.4f}')

    average_precision_train = sum(train_precision) / num_folds
    print(f'Average Precision Train K-Fold: {average_precision_train:.4f}')

    average_recall_val = sum(val_recall) / num_folds
    print(f'Average Recall Val K-Fold: {average_recall_val:.4f}')

    average_recall_train = sum(train_recall) / num_folds
    print(f'Average Recall Train K-Fold: {average_recall_train:.4f}')
    # predicting on test and calculating the metics on test set
    y_pred = classifier.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy Test: {accuracy * 100:.2f}%')

    test_mcc = matthews_corrcoef(y_test, np.round(y_pred))
    print(f'MCC Test: {test_mcc:.4f}')

    test_f1 = precision_recall_fscore_support(y_test, np.round(y_pred), average='weighted')[2]
    print(f'F1 Score Test: {test_f1:.4f}')

    test_precision, test_recall, _, _ = precision_recall_fscore_support(y_test, np.round(y_pred), average='weighted')
    print(f'Precision Test: {test_precision:.4f}')
    print(f'Recall Test: {test_recall:.4f}')

    class_report = classification_report(y_test, y_pred)
    print('Classification Report:')
    print(class_report)

    conf_matrix_test = confusion_matrix(y_test, np.round(y_pred))
    print('Confusion Matrix Test:')
    print(conf_matrix_test)
    # saving the model and writing the output to files
    if save_path and model_name:
        output_filename = os.path.join(save_path, f"{model_name}_outputs.txt")
        with open(output_filename, 'w') as f:
            f.write(f'Average Accuracy Val K-Fold: {average_accuracy_val * 100:.4f}\n')
            f.write(f'Average Accuracy Train K-Fold: {average_accuracy_train * 100:.4f}\n')
            f.write(f'Average MCC Val K-Fold: {average_mcc_val:.4f}\n')
            f.write(f'Average MCC Train K-Fold: {average_mcc_train:.4f}\n')
            f.write(f'Average F1 Score Val K-Fold: {average_f1_val:.4f}\n')
            f.write(f'Average F1 Score Train K-Fold: {average_f1_train:.4f}\n')
            f.write(f'Average Precision Val K-Fold: {average_precision_val:.4f}\n')
            f.write(f'Average Precision Train K-Fold: {average_precision_train:.4f}\n')
            f.write(f'Average Recall Val K-Fold: {average_recall_val:.4f}\n')
            f.write(f'Average Recall Train K-Fold: {average_recall_train:.4f}\n')
            f.write(f'Accuracy Test: {accuracy * 100:.2f}%\n')
            f.write(f'MCC Test: {test_mcc:.4f}\n')
            f.write(f'F1 Score Test: {test_f1:.4f}\n')
            f.write(f'Precision Test: {test_precision:.4f}\n')
            f.write(f'Recall Test: {test_recall:.4f}\n')
            f.write('Classification Report:\n')
            f.write(class_report + '\n')
            f.write('Confusion Matrix Test:\n')
            f.write(str(conf_matrix_test) + '\n')
        print(f"Outputs saved to: {output_filename}")
        joblib.dump(classifier, os.path.join(save_path, f"{model_name}.joblib"))
   
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, matthews_corrcoef, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import os
import joblib
import matplotlib.pyplot as plt

# function for stratified k-fold 
def evaluate_classifier_with_stratified_kfold(X_train, y_train, X_test, y_test, classifier, num_folds=10, save_path=None, model_name=None):
    # Stratified K-Fold
    k_fold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=0)
    # lists to store the values
    accuracies_val = []
    accuracies_train = []
    train_scores = []
    val_scores = []
    train_mcc = []
    val_mcc = []
    train_f1 = []
    val_f1 = []
    train_precision = []
    val_precision = []
    train_recall = []
    val_recall = []
    # scaling the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # running the stratified kfold
    for fold_idx, (train_indices, val_indices) in enumerate(k_fold.split(X_train_scaled, y_train), 1):
        X_fold_train, X_fold_val = X_train_scaled[train_indices], X_train_scaled[val_indices]
        y_fold_train, y_fold_val = y_train.iloc[train_indices], y_train.iloc[val_indices]
        # training and calculating the metrics
        try:
            classifier.fit(X_fold_train, y_fold_train)
            train_score = accuracy_score(y_fold_train, classifier.predict(X_fold_train))
            train_scores.append(train_score)
            y_train_pred = classifier.predict(X_fold_train)
            fold_accuracy_train = accuracy_score(y_fold_train, np.round(y_train_pred))
            accuracies_train.append(fold_accuracy_train)

            # Additional metrics: MCC, F1 Score, Precision, Recall
            train_mcc.append(matthews_corrcoef(y_fold_train, np.round(y_train_pred)))
            train_f1.append(precision_recall_fscore_support(y_fold_train, np.round(y_train_pred), average='weighted')[2])
            precision, recall, _, _ = precision_recall_fscore_support(y_fold_train, np.round(y_train_pred), average='weighted')
            train_precision.append(precision)
            train_recall.append(recall)

            val_score = accuracy_score(y_fold_val, classifier.predict(X_fold_val))
            val_scores.append(val_score)

            y_val_pred = classifier.predict(X_fold_val)
            fold_accuracy_val = accuracy_score(y_fold_val, np.round(y_val_pred))
            accuracies_val.append(fold_accuracy_val)

            # Additional metrics for validation set: MCC, F1 Score, Precision, Recall
            val_mcc.append(matthews_corrcoef(y_fold_val, np.round(y_val_pred)))
            val_f1.append(precision_recall_fscore_support(y_fold_val, np.round(y_val_pred), average='weighted')[2])
            precision, recall, _, _ = precision_recall_fscore_support(y_fold_val, np.round(y_val_pred), average='weighted')
            val_precision.append(precision)
            val_recall.append(recall)
        except Exception as e:
            print(f"Error during training: {e}")
            raise

    # Averaging the metrics

    average_accuracy_val = sum(accuracies_val) / num_folds
    print(f'Average Accuracy Val K-Fold: {average_accuracy_val * 100:.4f}')

    average_accuracy_train = sum(accuracies_train) / num_folds
    print(f'Average Accuracy Train K-Fold: {average_accuracy_train * 100:.4f}')

    average_mcc_val = sum(val_mcc) / num_folds
    print(f'Average MCC Val K-Fold: {average_mcc_val:.4f}')

    average_mcc_train = sum(train_mcc) / num_folds
    print(f'Average MCC Train K-Fold: {average_mcc_train:.4f}')

    average_f1_val = sum(val_f1) / num_folds
    print(f'Average F1 Score Val K-Fold: {average_f1_val:.4f}')

    average_f1_train = sum(train_f1) / num_folds
    print(f'Average F1 Score Train K-Fold: {average_f1_train:.4f}')

    average_precision_val = sum(val_precision) / num_folds
    print(f'Average Precision Val K-Fold: {average_precision_val:.4f}')

    average_precision_train = sum(train_precision) / num_folds
    print(f'Average Precision Train K-Fold: {average_precision_train:.4f}')

    average_recall_val = sum(val_recall) / num_folds
    print(f'Average Recall Val K-Fold: {average_recall_val:.4f}')

    average_recall_train = sum(train_recall) / num_folds
    print(f'Average Recall Train K-Fold: {average_recall_train:.4f}')
    # predicting on test set and calculating the metrics
    try:
        y_pred = classifier.predict(X_test_scaled)
    except Exception as e:
        print(f"Error during prediction on test set: {e}")
        raise

    try:
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy Test: {accuracy * 100:.2f}%')
    except Exception as e:
        print(f"Error calculating test set accuracy: {e}")
        raise

    try:
        test_mcc = matthews_corrcoef(y_test, np.round(y_pred))
        print(f'MCC Test: {test_mcc:.4f}')
    except Exception as e:
        print(f"Error calculating test set MCC: {e}")
        raise

    try:
        test_f1 = precision_recall_fscore_support(y_test, np.round(y_pred), average='weighted')[2]
        print(f'F1 Score Test: {test_f1:.4f}')
    except Exception as e:
        print(f"Error calculating test set F1 Score: {e}")
        raise

    try:
        test_precision, test_recall, _, _ = precision_recall_fscore_support(y_test, np.round(y_pred), average='weighted')
        print(f'Precision Test: {test_precision:.4f}')
        print(f'Recall Test: {test_recall:.4f}')
    except Exception as e:
        print(f"Error calculating test set Precision and Recall: {e}")
        raise

    try:
        class_report = classification_report(y_test, y_pred)
        print('Classification Report:')
        print(class_report)
    except Exception as e:
        print(f"Error generating classification report: {e}")
        raise

    try:
        conf_matrix_test = confusion_matrix(y_test, np.round(y_pred))
        print('Confusion Matrix Test:')
        print(conf_matrix_test)
    except Exception as e:
        print(f"Error generating test set confusion matrix: {e}")
        raise
    # saving the model and writing the output to files
    if save_path and model_name:
        output_filename = os.path.join(save_path, f"{model_name}_outputs.txt")
        with open(output_filename, 'w') as f:
            f.write(f'Average Accuracy Val K-Fold: {average_accuracy_val * 100:.4f}\n')
            f.write(f'Average Accuracy Train K-Fold: {average_accuracy_train * 100:.4f}\n')
            f.write(f'Average MCC Val K-Fold: {average_mcc_val:.4f}\n')
            f.write(f'Average MCC Train K-Fold: {average_mcc_train:.4f}\n')
            f.write(f'Average F1 Score Val K-Fold: {average_f1_val:.4f}\n')
            f.write(f'Average F1 Score Train K-Fold: {average_f1_train:.4f}\n')
            f.write(f'Average Precision Val K-Fold: {average_precision_val:.4f}\n')
            f.write(f'Average Precision Train K-Fold: {average_precision_train:.4f}\n')
            f.write(f'Average Recall Val K-Fold: {average_recall_val:.4f}\n')
            f.write(f'Average Recall Train K-Fold: {average_recall_train:.4f}\n')
            f.write(f'Accuracy Test: {accuracy * 100:.2f}%\n')
            f.write(f'MCC Test: {test_mcc:.4f}\n')
            f.write(f'F1 Score Test: {test_f1:.4f}\n')
            f.write(f'Precision Test: {test_precision:.4f}\n')
            f.write(f'Recall Test: {test_recall:.4f}\n')
            f.write('Classification Report:\n')
            f.write(class_report + '\n')
            f.write('Confusion Matrix Test:\n')
            f.write(str(conf_matrix_test) + '\n')
        print(f"Outputs saved to: {output_filename}")
        joblib.dump(classifier, os.path.join(save_path, f"{model_name}.joblib"))
