import numpy as np
import pandas as pd
import math
from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge
from sklearn import preprocessing
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import  SimpleImputer ,KNNImputer, IterativeImputer


def season2bool(season_vector):

    n_samples = len(season_vector)
    season_extension = np.zeros((n_samples,4))

    season_idx_dict = {
        'summer': 0,
        'autumn': 1,
        'winter': 2,
        'spring': 3
    }

    for i in range(len(season_vector)):
        s = season_vector[i]
        season_idx = season_idx_dict[s]
        season_extension[i,season_idx] = 1

    return season_extension

def removeMissingOutput(set):
    set = np.array(set)
    output = []

    for row in set:
        if math.isnan(row[0]):
            continue
        output.append(row)
            
    #print("\nRemoved ",counter_missing_y," line for missing y")
    return output

def crossValidation(X,y, kernelType, n_splits=10, alpha=1, gamma=1, normalize=False):

    if kernelType not in {'linear','polynomial','rbf','laplacian'}:
        print("Error: Invalid Kernel selection")
        return

    RMSE_vector = np.zeros((n_splits, 1))
    kf = KFold(n_splits=n_splits)
    split_counter = 0

    for train, test in kf.split(X):
        X_train, X_test, y_train, y_test = np.array(X[train]), np.array(X[test]), np.array(y[train]), np.array(y[test])

        # Normalization
        if normalize==True:
            X_train = np.concatenate((X_train,np.ones((X_train.shape[0],1))),axis=1)
            X_train = preprocessing.normalize(X_train,norm='l2',axis=1)
            X_test = np.concatenate((X_test,np.ones((X_test.shape[0],1))),axis=1)
            X_test = preprocessing.normalize(X_test,norm='l2',axis=1)


        # Kernel-Ridge regression
        krr = KernelRidge(alpha=alpha, gamma=gamma, kernel=kernelType)
        krr.fit(X_train,y_train)
        y_est = krr.predict(X_test)
        error = np.linalg.norm(y_est-y_test)
        RMSE_vector[split_counter] = error

        split_counter+=1

    avg_RMSE = np.mean(RMSE_vector)
    return avg_RMSE

    
def data_loading():

    imputation = "knn" # mean | knn | iterative
    neighbors = 6

    print("\nImputation:",imputation)
    if imputation!="mean":
        print("Neighbors:",neighbors)


    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """

    # Load training data
    train_df = pd.read_csv("train.csv")
    
    # Load test data
    test_df = pd.read_csv("test.csv")


    imputer_dict = {
        "mean" : SimpleImputer(missing_values=math.nan, strategy='mean'),
        "knn" : KNNImputer(n_neighbors=neighbors),
        "iterative" : IterativeImputer(missing_values=math.nan, initial_strategy='mean')
    }

    # Train imputation
    imputer_train = imputer_dict[imputation]
    train_set = np.array(train_df)

    train_set_onlyOutput = train_set[:,2]
    train_set_onlySeason = train_set[:,0]

    train_set_noSeason = train_set[:,1:]
    train_set_onlyX =  np.delete(train_set_noSeason,1,1)
    train_set_onlyX = imputer_train.fit_transform(train_set_onlyX)

    train_set = np.concatenate((train_set_onlyOutput.reshape((len(train_set_onlyOutput), 1)),season2bool(train_set_onlySeason),train_set_onlyX),axis=1)                       
    train_set = np.array(removeMissingOutput(train_set))

    X_train = np.delete(train_set,0,1)
    y_train = train_set[:,0]

    imputer_test = imputer_dict[imputation]
    test_set = np.array(test_df)
    test_set_onlySeason = test_set[:,0]
    test_set_onlyX = test_set[:,1:]
    test_set_onlyX = imputer_test.fit_transform(test_set_onlyX)
    X_test = np.concatenate((season2bool(test_set_onlySeason),test_set_onlyX),axis=1)  

    

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"

    return X_train,y_train,X_test

def modeling_and_prediction(X_train, y_train, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data 

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """

    # Define test cases
    results = {}
    kernels = ['rbf','laplacian']
    alphas = [ 0.01, 0.02, 0.03, 0.04]
    gammas = [0.01, 0.1, 1]
    
    for k in kernels:
        for a in alphas:
            for g in gammas:
                # Apply cross validation
                results[(k,a,g)] = crossValidation(X_train,y_train,alpha=a, gamma=g, normalize=False,kernelType=k)


    # Select minimum RMSE method
    best_method = min(results, key=results.get)
    print("Best kernel: ",best_method)
    print("Score:",results[best_method])


    # Predict output
    kernel = best_method[0]
    alpha = best_method[1]
    gamma = best_method[2]

    krr = KernelRidge(kernel=kernel, alpha=alpha, gamma=gamma)
    krr.fit(X_train,y_train)
    y_pred = krr.predict(X_test)

    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred



# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    # The function retrieving optimal LR parameters
    y_pred=modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

