from CellularModel import CellularModel
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score
import multiprocessing
import numpy as np

def log_trial(target_log='tests/log2.json', dataset='iris'):
    '''
    Runs an instance on a sample dataset

    Expects:
        target_log          STR destination filepath of JSON log
        dataset             STR scikit dataset used for comparison
    '''

    assert isinstance(target_log,str),\
        'Expected target_log STR, got type {}'.format(type(target_log))
    assert isinstance(dataset, str), \
        'Expected dataset STR, got type {}'.format(type(dataset))
    assert dataset in ['iris'],\
        'Unrecognizes dataset= "{}"'.format(dataset)

    if dataset == 'iris':
        X, y = load_iris(return_X_y=True)
        y = np.array(y).reshape(150, 1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30)

    # Train on training set
    model = CellularModel(fitness_func='accuracy',
                          N=100,
                          mutate_proba=0.001,
                          min_depth=20,
                          max_depth=20,
                          max_epochs=45,
                          auto_plot=True,
                          experimental_mode=True,
                          components=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test, plot=True)
    f1 = f1_score(y_test, y_pred, average='macro')
    print('The f1_score was {}'.format(f1))

    return

def start_trials(numTrials=7):
    '''
    Starts a number of multiprocessed trials

    Expects:
        numTrials           INT the number of instances to run
    '''

    jobs = []
    for i in range(numTrials):
        p = multiprocessing.Process(target=log_trial)
        jobs.append(p)
        p.start()

    return
