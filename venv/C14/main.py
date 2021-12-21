#for laoding as numpy array
import arff, numpy as np
#from scipy.io import arff
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import PredefinedSplit
from hypopt import GridSearch
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import plot_confusion_matrix, recall_score
from sklearn.utils import resample

'''
def add_test_labels():
    labels = pd.read_table('/home/local/Dokumente/CI_14/Snore_dist/lab/ComParE2017_Snore_test.tsv')
    #labels = pd.read_table('/home/local/Dokumente/CI_14/Cold_dist/lab/ComParE2017_Cold_with_test_labels.csv')
    data = pd.read_csv('Snore_Test_no_labels_deepspectrum.csv')
    data = data.rename(columns={data.columns[0]: 'file_name'})
    df_merged = pd.merge(data, labels, how='inner', on='file_name')
    print(df_merged)
    df_merged.to_csv('Snore_Test_merged_with_labels_deepspectrum.csv', index=False)

def grid_search_with_val():
    df_train = pd.read_csv('Cold_Train_merged_with_labels_deepspectrum.csv')
    df_devel = pd.read_csv('Cold_Devel_merged_with_labels_deepspectrum.csv')
    df_test = pd.read_csv('Cold_Test_merged_with_labels_deepspectrum.csv')
    print(df_train)

    X_train = df_train.iloc[:, 1:-2]
    print(X_train)
    y_train = df_train.iloc[:, -1:]
    print(y_train)
    X_devel = df_devel.iloc[:, 1:-2]
    print(X_devel)
    y_devel = df_devel.iloc[:, -1:]
    print(y_devel)
    X_test = df_test.iloc[:, 1:-1]
    print(X_test)
    y_test = df_test.iloc[:, -1:]
    print(y_test)

    param_grid = {
        'scaler__feature_range': [(-1, 1)],
        'dimensionality_reduction__n_components': [100],
        'classifier__C': [0.001]
        #'scaler__feature_range': [(0, 1), (-1, 1)],
        #'dimensionality_reduction__n_components': [8, 16, 32, 100, 200],
        #'classifier__C': np.logspace(0, -8, num=9)
    }

    pipeline = Pipeline(steps=[('scaler', MinMaxScaler()),
                               ('dimensionality_reduction', PCA()),
                               ('classifier', LinearSVC(max_iter=10000))])
    # Grid-search all parameter combinations using a validation set.
    opt = GridSearch(model=pipeline, param_grid=param_grid)
    opt.fit(X_train, y_train.values.ravel(), X_devel, y_devel)
    print(f'Best Estimator: {opt.best_estimator_}')
    print(f'Best Parameters: {opt.best_params}')
    print(f'Best Score on Devel: {opt.best_score}')
    print('Test Score for Optimized Parameters:', opt.score(X_test, y_test))
    preds = opt.predict(X_test)
    print(type(opt))
    print (classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds, labels=["C", "NC"]))
    disp = plot_confusion_matrix(opt, X_devel, y_devel,
                                 display_labels=["C", "NC"],
                                 cmap=plt.cm.Blues,
                                 normalize=None)
    # disp.ax_.set_title(title)
    print(disp.confusion_matrix)
    plt.show()
    plt.savefig('confusion_matrix_Snore_deepspectrum.png')
'''

def merge_data():
    # load as numpy array
    dataset = arff.load(open('/home/local/Dokumente/HeartApp/Feature_Extraction/All_Data_splitted_training_DS_VGG19.arff', 'r'))
    data = np.array(dataset['data'])
    print(data)

    # for loading as pandas datafram
    #dataset = arff.loadarff(open('/home/local/Dokumente/HeartApp/Feature_Extraction/All_Data_splitted_training_DS_VGG19.arff', 'r'))
    data = pd.DataFrame(dataset)
    print(type(data))
    print(data)

    colnames = ['filename', 'labels']
    labels = pd.read_csv('/home/local/Dokumente/HeartApp/training_labels_all_data_splitted_combined.csv', names=colnames, header=None)
    print(labels)
    data = data.rename(columns={0: 'filename'})
    data['filename'] = data['filename'].replace(".wav", "", regex=True)
    print(f"After:{data}")

    df_merged = data.merge(labels, 'inner')
    print(df_merged)
    df_merged.to_csv('/home/local/Dokumente/HeartApp/Feature_Extraction/All_Data_splitted_training_DS_VGG19.csv', index=False)




def train_model():

    df_train = pd.read_csv('Cold_Train_merged_with_labels_vgg16_None.csv')
    df_devel = pd.read_csv('Cold_Devel_merged_with_labels_vgg16_None.csv')
    df_test = pd.read_csv('Cold_Test_merged_with_labels_vgg16_None.csv')
    df_train = pd.concat([df_train, df_devel], ignore_index=True)
    print(df_train)

    X_train = df_train.iloc[:,1:-1]
    print(X_train)
    y_train = df_train.iloc[:,-1:]
    print(y_train)

    X_test = df_test.iloc[:,1:-1]
    print(X_test)
    y_test = df_test.iloc[:, -1:]
    print(y_test)

    print(df_train['label'].value_counts())
    max = df_train['label'].value_counts().max()
    normal = df_train[df_train.filename == -1]
    print(normal)
    abnormal = df_train[df_train.filename == 1]

    abnormal = resample(abnormal, replace=True, n_samples=max, random_state=120)

    df_train = pd.concat([normal, abnormal], ignore_index=True)
    print(df_train['label'].value_counts())
    print(df_train)


    param_grid = {
            'scaler__feature_range': [(0, 1)],
            'dimensionality_reduction__n_components': [100],
            'classifier__C': [0.1]
        }

    pipeline = Pipeline(steps=[('scaler', MinMaxScaler()),
    ('dimensionality_reduction', PCA()),
    ('classifier', LinearSVC(max_iter=10000,class_weight='balanced'))])


    gs = GridSearchCV(pipeline, cv=2, scoring="recall_macro", refit=True, param_grid=param_grid,
                      verbose=2)
    gs.fit(X_train, y_train.values.ravel())
    print(f'Best Train score: {gs.best_score_}')
    preds_test = gs.predict(X_test)
    print(f"Test Score: {recall_score(y_test, preds_test, average='macro')}")
    # print(f"Best Test Score: {gs.best_estimator_.score(X_test, y_test)}")
    print(f'Best Parameters: {gs.best_params_}')


    print(classification_report(y_test, preds_test))
    #disp = plot_confusion_matrix(gs, X_test, y_test, display_labels=["E", "O", "T", "V"], cmap=plt.cm.Blues, normalize=None)
    disp = plot_confusion_matrix(gs, X_test, y_test, display_labels=["C", "NC"], cmap=plt.cm.Blues, normalize=None)
    disp.ax_.set_title('Cold Deepspectrum VGG16 Random Upsampled')
    print(disp.confusion_matrix)
    plt.savefig('conf_matrix_cold_vgg16_random_upsampled.png')
    plt.show()





def grid_search():
    df_train = pd.read_csv('/home/local/Dokumente/HeartApp/physionet_challenge/Physionet_Train_merged_with_labels_VGG16.csv')
    df_test = pd.read_csv('/home/local/Dokumente/HeartApp/physionet_challenge/Physionet_validation_merged_with_labels_VGG16.csv')
    print(df_train)
    print(df_test)
    df_train = df_train[~df_train.file_name.isin(df_test['file_name'])]


    X_train = df_train.iloc[:, 1:-1]
    print(X_train)
    y_train = df_train.iloc[:, -1:]
    print(y_train)
    X_test = df_test.iloc[:, 1:-1]
    print(X_test)
    y_test = df_test.iloc[:, -1:]
    print(y_test)

    '''
     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}] SVC()
    '''

    pipelines = [
        Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('classifier', LinearSVC(max_iter=10000,class_weight='balanced'))
            ]
        ),
        Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('dimensionality_reduction', PCA()),
                ('classifier', LinearSVC(max_iter=10000, class_weight='balanced'))
            ]
        ),
        Pipeline(
            steps=[
                ('scaler', MinMaxScaler()),
                ('dimensionality_reduction', PCA()),
                ('classifier', LinearSVC(max_iter=10000, class_weight='balanced'))
            ]
        )
    ]

    parameter_grids = [
        {
            'classifier__C': [0.001, 0.01, 0.1, 1.0]
        },
        {
            'dimensionality_reduction__n_components': [100, 200],
            'classifier__C': [ 0.001, 0.01, 0.1, 1.0]
        },
        {
            'scaler__feature_range': [(0, 1), (-1, 1)],
            'dimensionality_reduction__n_components': [100, 200],
            'classifier__C': [0.001, 0.01, 0.1, 1.0]
        }
    ]
    scores_best_estimators = []
    params_best_estimators = []
    best_estimators = []
    best_scores = []
    for i in range(len(pipelines)):
        gs = GridSearchCV(pipelines[i], parameter_grids[i], scoring='recall_macro', cv=8, verbose=2)
        gs.fit(X_train, y_train.values.ravel())
        best_scores.append(gs.best_score_)
        scores_best_estimators.append(gs.best_estimator_.score(X_test, y_test))
        params_best_estimators.append(gs.best_params_)
        best_estimators.append(gs.best_estimator_)

    print(best_scores)
    print(scores_best_estimators)
    print(params_best_estimators)
    print(best_estimators)



if __name__ == '__main__':
    merge_data()
    #train_model()
    #grid_search()


