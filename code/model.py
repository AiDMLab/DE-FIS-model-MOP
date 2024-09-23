import geatpy as ea
from multiprocessing import Pool as ProcessPool
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_absolute_percentage_error
from sklearn.metrics import r2_score,make_scorer



class MyProblem(ea.Problem):
    def __init__(self, PoolType):
        name = 'MyProblem'
        M = 1
        maxormins = [-1]
        Dim = 5
        varTypes = [1] * Dim
        lb = [100, 2, 2, 10, 2]
        ub = [600, 50, 50, 1000, 10]
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
        self.data = X_train
        self.dataTarget = y_train
        self.PoolType = PoolType
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(2)
        elif self.PoolType == 'Process':
            num_cores = int(mp.cpu_count())
            self.pool = ProcessPool(num_cores)

    def evalVars(self, Vars):
        N = Vars.shape[0]
        args = list(zip(list(range(N)), [Vars] * N, [self.data] * N, [self.dataTarget] * N))
        if self.PoolType == 'Thread':
            f = np.array(list(self.pool.map(subAimFunc, args)))
        elif self.PoolType == 'Process':
            result = self.pool.map_async(subAimFunc, args)
            result.wait()
            f = np.array(result.get())
        return f


def subAimFunc(args):
    i = args[0]
    Vars = args[1]
    data = args[2]
    dataTarget = args[3]

    steps = list()
    steps.append(('scaler', MinMaxScaler()))
    steps.append(('power', PowerTransformer()))
    steps.append(('model', GradientBoostingRegressor(
        n_estimators=int(Vars[i, 0]),
        learning_rate=Vars[i, 1],
        max_depth=int(Vars[i, 2]),
        min_samples_split=int(Vars[i, 3]),
        min_samples_leaf=int(Vars[i, 4]),
        random_state=42)))
    pipeline = Pipeline(steps=steps)
#model=svr,rfr,gbr,abr,mlp

    # prepare the model with target scaling
    model = TransformedTargetRegressor(regressor=pipeline, transformer=PowerTransformer())
    model.fit(data, dataTarget)
    scores = cross_val_score(model, data, dataTarget, scoring='r2', cv=10)
    ObjV_i = [scores.mean()]
    return ObjV_i


def main():
    # Load data
    data = pd.read_csv('')
    X = data.iloc[:, ] #features
    y = data.iloc[:, ] #target


    # Split data into train and test sets
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create problem instance
    problem = MyProblem(PoolType='Thread')

    # Run the optimization
    algorithm = ea.soea_DE_best_1_bin_templet(
        problem,
        ea.Population(Encoding='RI', NIND=20),
        MAXGEN=30,
        logTras=1,
        trappedValue=1e-6,
        maxTrappedCount=10
    )

    res = ea.optimize(algorithm,
                      verbose=True,
                      drawing=1,
                      outputMsg=True,
                      drawLog=False,
                      saveFlag=True)

    # Get the best parameters from the optimization result
    best_params = res['Vars'][0]  # Assuming the best individual is the first one

    # Train final model with the best parameters
    final_model = GradientBoostingRegressor(
        n_estimators=int(best_params[0]),
        learning_rate=best_params[1],
        max_depth=int(best_params[2]),
        min_samples_split=int(best_params[3]),
        min_samples_leaf=int(best_params[4]),
        random_state=42
    )

    # Prepare the model with target scaling
    final_pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('power', PowerTransformer()),
        ('model', final_model)
    ])

    final_transformed_model = TransformedTargetRegressor(regressor=final_pipeline, transformer=PowerTransformer())
    final_transformed_model.fit(X_train, y_train)

    # evaluate model
    cv = KFold(n_splits=10, random_state=None)
    R2 = make_scorer(r2_score)
    RMSE = make_scorer(mean_absolute_error)
    results = []
    scores = cross_val_score(final_transformed_model, X_train, y_train, scoring=R2, cv=cv, n_jobs=-1)
    scores = np.absolute(scores)
    # _mean = np.mean(scores)
    results.append(scores)

    # Evaluate the model on the test set
    y_train_pred = final_transformed_model.predict(X_train)
    y_pred = final_transformed_model.predict(X_test)
    r2_score_test = r2_score(y_test, y_pred)
    r2_score_train = r2_score(y_train, y_train_pred)
    RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred))
    RMSE_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    #print("Final R2 score on the test set:", r2_score_test)

    # 保存模型
    # dump(model, open('../model_pickle/ ', 'wb'))


if __name__ == '__main__':
    main()
