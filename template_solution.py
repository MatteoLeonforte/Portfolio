# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.kernel_ridge import KernelRidge

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge, LinearRegression, RidgeCV
from sklearn.metrics import accuracy_score




import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MY FUNCTIONS
def save_features(array: np.ndarray,name:str):
    np.savetxt(name, array, delimiter=',')

def load_features() -> np.ndarray:
    return np.genfromtxt('features.csv', delimiter=',')
def regression_cv(X,y,regressor, normalize=False,scaler=MinMaxScaler()):

    score_tr_vector = np.zeros((10, 1))
    score_val_vector = np.zeros((10, 1))
    kf = KFold(n_splits=10)
    split_counter = 0

    for train, test in kf.split(X):
        x_feat_tr, x_feat_val, y_feat_tr, y_feat_val = np.array(X[train]), np.array(X[test]), np.array(y[train]), np.array(y[test])
    
        
        regressor = regressor.fit(x_feat_tr,y_feat_tr)
        score_tr = regressor.score(x_feat_tr,y_feat_tr)
        score_val = regressor.score(x_feat_val,y_feat_val)
        score_tr_vector[split_counter] = score_tr
        score_val_vector[split_counter] = score_val
        split_counter+=1

    avg_tr_score = np.mean(score_tr_vector)
    avg_val_score = np.mean(score_val_vector)

    return avg_tr_score,avg_val_score

def create_loader_from_np(X, y = None, train = True, batch_size=None, shuffle=True, num_workers = 2):
    """
    Create a torch.utils.data.DataLoader object from numpy arrays containing the data.

    input: X: numpy array, the features
           y: numpy array, the labels
    
    output: loader: torch.data.util.DataLoader, the object containing the data
    """
    if train:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float), 
                                torch.from_numpy(y).type(torch.float))
    else:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float))
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=True, num_workers=num_workers)
    return loader

# END MY FUNCTIONS


def load_data():
    """
    This function loads the data from the csv files and returns it as numpy arrays.

    input: None
    
    output: x_pretrain: np.ndarray, the features of the pretraining set
            y_pretrain: np.ndarray, the labels of the pretraining set
            x_train: np.ndarray, the features of the training set
            y_train: np.ndarray, the labels of the training set
            x_test: np.ndarray, the features of the test set
    """
    x_pretrain = pd.read_csv("public/pretrain_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_pretrain = pd.read_csv("public/pretrain_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_train = pd.read_csv("public/train_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_train = pd.read_csv("public/train_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_test = pd.read_csv("public/test_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1)
    return x_pretrain, y_pretrain, x_train, y_train, x_test

class Net(nn.Module):
    """
    The model class, which defines our feature extractor used in pretraining.
    """
    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        # TODO: Define the architecture of the model. It should be able to be trained on pretraing data 
        # and then used to extract features from the training and test data.
        input_size = 1000
        hid1 = 1600
        hid2 = 2048
        hid3 = 1600
        hid4 = 512

        self.fc1 = nn.Linear(input_size, hid1)
        self.act1 = nn.Tanh()
        self.norm1 = nn.BatchNorm1d(hid1,affine=False)
        self.drop1 = nn.Dropout(p=0.3)
        
        self.fc2 = nn.Linear(hid1, hid2)
        self.act2 = nn.LeakyReLU()
        self.norm2 = nn.BatchNorm1d(hid2,affine=False)
        self.drop2 = nn.Dropout(p=0.3)
        
        
        
        self.fc3 = nn.Linear(hid2,hid3)
        self.act3 = nn.LeakyReLU()
        self.norm3 = nn.BatchNorm1d(hid3,affine=False)
        self.drop3 = nn.Dropout(p=0.3)

        self.features = nn.Linear(hid3, hid4)
        self.act4 = nn.Sigmoid()

        self.output = nn.Linear(hid4, 1)
        
        
        

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """

        
        x = self.fc1(x)
        x = self.act1(x)  
        x = self.norm1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.act2(x) 
        x = self.norm2(x)
        x = self.drop2(x)

        x = self.fc3(x)
        x = self.act3(x) 
        x = self.drop3(x)
        x = self.norm3(x)


        x = self.features(x)
        x = self.act4(x) 
    
        x = self.output(x)

        return x
    
def make_feature_extractor(x_in, y_in, batch_size=256, eval_size=1000, normalize=False, scaler = MinMaxScaler()):
    """
    This function trains the feature extractor on the pretraining data and returns a function which
    can be used to extract features from the training and test data.

    input: x: np.ndarray, the features of the pretraining set
              y: np.ndarray, the labels of the pretraining set
                batch_size: int, the batch size used for training
                eval_size: int, the size of the validation set
            
    output: make_features: function, a function which can be used to extract features from the training and test data
    """
    # Pretraining data loading
    x_tr, x_val, y_tr, y_val = train_test_split(x_in, y_in, test_size = eval_size, random_state = 1, shuffle = True)

    
    # create train and validation loader
    train_loader =  create_loader_from_np(x_tr,  y_tr,  train = True, batch_size = batch_size)
    val_loader =    create_loader_from_np(x_val, y_val, train = True, batch_size = batch_size)

    # Normalize
    if normalize:
        x_tr = scaler.fit_transform(x_tr)
        x_val = scaler.fit_transform(x_val)

    # model declaration
    model = Net()
    model.to(device=device)
    n_epochs = 100
    lr = 0.001
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr) 
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.01, total_iters=n_epochs/2)
    
    # Training loop
    train_loss = []
    val_loss = []

    for epoch in range(n_epochs):
        
        model.train()
        # Train
        train_batch_loss = []
        for (X, y) in train_loader:

            X = X.to(device=device)
            y = y.to(device=device)

            # Forward
            y_hat = model(X).squeeze()
            loss = criterion(y_hat, y)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()

            # GD
            optimizer.step()

            train_batch_loss.append(loss.item())
            
        train_epoch_loss = np.mean(train_batch_loss)
        train_loss.append(train_epoch_loss)

        scheduler.step()

        # Validation
        model.eval()

        val_batch_loss = []
        for (X, y) in val_loader:
            #X = torch.tensor(x_val, dtype=torch.float)
            #y = torch.tensor(y_val, dtype=torch.float)

            X = X.to(device=device)
            y = y.to(device=device)

            # Forward
            y_hat = model(X).squeeze()
            loss = criterion(y, y_hat)        


            val_batch_loss.append(loss.item())


        val_epoch_loss = np.mean(val_batch_loss)
        val_loss.append(val_epoch_loss)


        print(f"Epoch {epoch}   ||  Train: {train_epoch_loss}    ||  Eval:   {val_epoch_loss} || Lr: {optimizer.param_groups[0]['lr']}")
        
    plt.plot(range(n_epochs),train_loss,'r',range(n_epochs),val_loss,'b')
    plt.legend(["Train lost","Validation lost"])
    plt.show()

    # Complete Pretraining
    #### Training on the whole pretraining

    feature_extractor = torch.nn.Sequential(*list(model.children())[:-2])
    return feature_extractor
    """
    def make_features(x,feature_size,normalize=False, scaler=MinMaxScaler()):
        
        This function extracts features from the training and test data, used in the actual pipeline 
        after the pretraining.

        input: x: np.ndarray, the features of the training or test set

        output: features: np.ndarray, the features extracted from the training or test set, propagated
        further in the pipeline
        

        if normalize:
            x = scaler.fit_transform(x)

        model.eval()
        layer = model._modules.get('features')
        extracted_features = torch.zeros((len(x),feature_size))

        def copy_data(m, i, o):
            extracted_features.copy_(o.data.squeeze())
        
        h = layer.register_forward_hook(copy_data)
        x = torch.tensor(x, dtype=torch.float)
        model(x)
        h.remove()

        

        return extracted_features

    return make_features,feature_extractor          """

def make_pretraining_class(feature_extractors):
    """
    The wrapper function which makes pretraining API compatible with sklearn pipeline
    
    input: feature_extractors: dict, a dictionary of feature extractors

    output: PretrainedFeatures: class, a class which implements sklearn API
    """

    class PretrainedFeatures(BaseEstimator, TransformerMixin):
        """
        The wrapper class for Pretraining pipeline.
        """
        def __init__(self, *, feature_extractor=None, mode=None):
            self.feature_extractor = feature_extractor
            self.mode = mode

        def fit(self, X=None, y=None):
            return self

        def transform(self, X):
            assert self.feature_extractor is not None
            X_new = feature_extractors[self.feature_extractor](X)
            return X_new
        
    return PretrainedFeatures

def get_regression_model(regr_name:str, alpha:float, gamma:float):
    """
    This function returns the regression model used in the pipeline.

    input: None

    output: model: sklearn compatible model, the regression model
    """

    # TODO: Implement the regression model. It should be able to be trained on the features extracted
    # by the feature extractor.
    regressors = {
        'lin': LinearRegression(),
        'ridge': Ridge(alpha=alpha, fit_intercept=False),
        'rbf': KernelRidge(alpha=alpha, gamma=gamma, kernel='rbf')
    }
    
    return regressors[regr_name]

# Main function. You don't have to change this
if __name__ == '__main__':
    # Load data
    x_pretrain, y_pretrain, x_train, y_train, x_test = load_data()
    print("Data loaded!")
    
    # Normalization
    normalize = False
    scaler = MinMaxScaler()

    feature_size = 512

    existing_features = False
    
    if not existing_features:
        print("Recomputing features")
        # Pretraining
        feature_extractor =  make_feature_extractor(x_pretrain, y_pretrain,normalize=normalize, scaler=scaler)
        PretrainedFeatureClass = make_pretraining_class({"pretrain": feature_extractor})

        # Get features from X_train and X_test
        input_train = torch.tensor(x_train, dtype=torch.float).to(device=device)
        X_train_features = feature_extractor(input_train)
        print(f"X_train_features extracted : {X_train_features.shape}")
        assert(X_train_features.shape[1]==feature_size)

        input_test = torch.tensor(x_test.to_numpy(), dtype=torch.float).to(device=device)
        X_test_features = feature_extractor(input_test)
        assert(X_test_features.shape[1]==feature_size)
        print(f"X_test_features extracted : {X_test_features.shape}")

        X_train_features = X_train_features.cpu().detach().numpy() 
        X_test_features = X_test_features.cpu().detach().numpy() 
        save_features(X_train_features,'X_train_features.csv')
        save_features(X_test_features,'X_test_features.csv')

    else:
        X_train_features = load_features('X_train_features.csv')
        X_test_features = load_features('X_test_features.csv')

    y_pred = np.zeros(x_test.shape[0])

    regressor = RidgeCV(np.arange(start=0.0001,stop=10,step=0.001),fit_intercept=False,cv=10)
    X_train_features_tr, X_train_features_val, y_train_tr, y_train_val = train_test_split(X_train_features,0.3)
    regressor.fit(X_train_features_tr,y_train_tr)
    print(f"Score regression: {regressor.score(X_train_features_val,y_train_val)}")
    y_pred = regressor.predict(X_test_features)
    
    assert y_pred.shape == (x_test.shape[0],)
    y_pred = pd.DataFrame({"y": y_pred}, index=x_test.index)
    y_pred.to_csv("results.csv", index_label="Id")
    print("Predictions saved, all done!")

