import numpy as np
import pandas as pd
import geatpy as ea
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from multiprocessing import Pool as ProcessPool
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import ElasticNetCV,ElasticNet
from sklearn.metrics import mean_squared_error, r2_score



data = pd.read_csv('../data_descriptor_select177.csv')
#data = pd.read_csv('../data_key_descriptor_select.csv')
X = data.iloc[:, 0:177]
y = data.iloc[:, 178]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
#fit scaler on the training dataset
scaler.fit(X_train)
#transform both datasets
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


class MyProblem(ea.Problem):
    def __init__(self, PoolType):
        name = 'MyProblem'
        M = 1
        maxormins = [-1]
        Dim = 177
        varTypes = [1] * Dim
        lb = [0] * Dim
        ub = [1] * Dim
        lbin = [1] * Dim
        ubin = [1] * Dim
        ea.Problem.__init__(self,
                            name,
                            M,
                            maxormins,
                            Dim,
                            varTypes,
                            lb,
                            ub,
                            lbin,
                            ubin)
        self.data = X_train_scaled
        self.dataTarget = y_train
        self.PoolType = PoolType
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(2)
        elif self.PoolType == 'Process':
            num_cores = int(mp.cpu_count())
            self.pool = ProcessPool(num_cores)

    def aimFunc(self, pop):
        Vars = pop.Phen
        fitness = []
        for i in range(Vars.shape[0]):
            args = (i, Vars[i], self.data, self.dataTarget)
            fitness.append(subAimFunc(args))
        pop.ObjV = np.array(fitness)

def subAimFunc(args):
    i = args[0]
    Vars = args[1]
    data = args[2]
    dataTarget = args[3]

    #选择K个最佳特征
    num_features = np.sum(Vars)
    if num_features == 0:
        return [-np.inf] #如果没有选择任何特征，则返回负无穷，表示该个体不可行

    #selector = SelectKBest(score_func=f_regression, k=num_features)
    #selected_data = selector.fit_transform(data, dataTarget)
    selected_data = data[:, Vars.astype(bool)]

    #进行回归建模

    steps = list()
    steps.append(('scaler', StandardScaler()))
    steps.append(('power', PowerTransformer()))
    steps.append(('model', GradientBoostingRegressor()))
    pipeline = Pipeline(steps=steps)

    # prepare the model with target scaling
    clf = TransformedTargetRegressor(regressor=pipeline, transformer=PowerTransformer())
    clf.fit(selected_data, y_train)
    scores = cross_val_score(clf, selected_data, dataTarget, cv=10, scoring='r2')
    avg_score = np.mean(scores)
    #clf.fit(selected_data, dataTarget)



    return [avg_score]

#创建问题对象
problem = MyProblem(PoolType='Thread')
# Run GA multiple times
num_runs = 10
best_subsets = []


for _ in range(num_runs):
    #problem = MyProblem(PoolType='Process')
    algorithm = ea.soea_DE_best_1_bin_templet(
        problem,
        ea.Population(Encoding='RI', NIND=200),
        MAXGEN=50,
        logTras=1,
        trappedValue=1e-6,
        maxTrappedCount=10
    )

    # Optimize
    res = ea.optimize(algorithm,
                      verbose=True,
                      drawing=1,
                      outputMsg=True,
                      drawLog=False,
                      saveFlag=True)

    best_individual = res['Vars'][0]
    selected_features = np.where(best_individual == 1)[0]
    best_subsets.append(selected_features)

# Determine the threshold for predictor frequency
threshold = num_runs * 0.8  # Adjust threshold as needed

# Count frequency of each predictor
predictor_frequency = {}
for subset in best_subsets:
    for predictor in subset:
        predictor_frequency[predictor] = predictor_frequency.get(predictor, 0) + 1

# Select predictors for the final subset based on frequency
final_subset = [predictor for predictor, freq in predictor_frequency.items() if freq >= threshold]

# Print final selected predictors
#print("Final Selected Features:", final_subset)

#使用梯度提升树模型进行特征重要性排序
gbr = GradientBoostingRegressor()
gbr.fit(X_train[:, final_subset], y_train)

y_pred = gbr.predict(X_test[:, final_subset])
R2 = r2_score(y_test, y_pred)
print("R2:", R2)

# 获取特征重要性
feature_importances = gbr.feature_importances_

# 根据特征重要性对特征进行排序
sorted_indices = np.argsort(feature_importances)[::-1]

# 选择重要性大于某个阈值的特征
importance_threshold = 0.01  # 可以根据需要调整阈值
selected_features_after_gbt = np.where(feature_importances > importance_threshold)[0]

print("Selected Features after GBT:", selected_features_after_gbt)

# 使用梯度提升树模型对所筛选特征进行建模
#gbrb = model.py
#gbr_b.fit(X_train[:, selected_features_after_gbt], y_train)

# 在测试集上评价模型
#y_pred_after_gbt = gbr_b.predict(X_test[:, selected_features_after_gbt])
#R2_after_gbt = r2_score(y_test, y_pred_after_gbt)

