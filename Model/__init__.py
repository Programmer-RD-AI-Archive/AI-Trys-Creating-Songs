from Model.data import *
from Model.help_funcs import *
import random
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

# Preproccessing
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    MaxAbsScaler,
    OneHotEncoder,
    Normalizer,
    Binarizer,
)




# Feature Selection
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel

# Model Eval
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
)

# Models
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    AdaBoostRegressor,
    VotingRegressor,
    BaggingRegressor,
    RandomForestRegressor,
)
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from catboost import CatBoost, CatBoostRegressor
from xgboost import XGBRegressor, XGBRFRegressor

# Other
import pickle
import wandb

PROJECT_NAME = "AI-Trys-Creating-Songs"
device = "cuda"
from Model.data import *
from Model.help_funcs import *
import random
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

# Preproccessing
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    MaxAbsScaler,
    OneHotEncoder,
    Normalizer,
    Binarizer,
)

# Decomposition
from sklearn.decomposition import (
    PCA,
    KernelPCA,
    DictionaryLearning,
    FastICA,
    IncrementalPCA,
    MiniBatchDictionaryLearning,
    MiniBatchSparsePCA,
    NMF,
    SparseCoder,
    SparsePCA,
    dict_learning,
    dict_learning_online,
    fastica,
    non_negative_factorization,
    randomized_svd,
    sparse_encode,
    FactorAnalysis,
    TruncatedSVD,
    LatentDirichletAllocation,
)
from scipy.linalg import svd

# Feature Selection
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel

# Model Eval
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
)

# Models
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    AdaBoostRegressor,
    VotingRegressor,
    BaggingRegressor,
    RandomForestRegressor,
)
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from catboost import CatBoost, CatBoostRegressor
from xgboost import XGBRegressor, XGBRFRegressor

# Other
import pickle
import wandb

PROJECT_NAME = "AI-Trys-Creating-Songs"
device = "cuda"
