import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,PowerTransformer
from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from scipy.stats import pearsonr

#数据
data = pd.read_csv('data_descriptor_select177.csv')

features1 = ['ave:covalent_radius_pyykko_double', 'ave:hhi_p', 'ave:num_s_unfilled',
       'sum:fusion_enthalpy', 'var:bulk_modulus', 'var:num_p_unfilled',
       'var:num_s_unfilled', 'var:num_s_valence', 'var:vdw_radius_uff', 'TT',
       'TT_time']
features = ['ave:covalent_radius_pyykko_double',
       'sum:fusion_enthalpy', 'var:bulk_modulus', 'TT', 'TT_time']
X = data[features1]
y = data.iloc[:, 178]
# 选择归一化方法
scaler = MinMaxScaler()

# 对数据进行归一化
X_norm = scaler.fit_transform(X)

power = PowerTransformer()
X_norm_power = power.fit_transform(X_norm)
# 创建并训练符号回归模型
est = SymbolicRegressor(
    population_size=10000,
    generations=50,
    stopping_criteria=0.01,
    p_crossover=0.8,
    p_subtree_mutation=0.1,
    p_hoist_mutation=0.05,
    p_point_mutation=0.05,
    max_samples=0.9,
    tournament_size=30,
    const_range=(-1, 1),
    verbose=1,
    parsimony_coefficient=0.1,
    random_state=42
)
est.fit(X_norm, y)

# 预测
y_pred = est.predict(X_norm)

# 计算误差
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y,y_pred)
pcc, _ = pearsonr(y, y_pred)

# 打印误差
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R2: {r2}")

# 可视化结果
#plt.scatter(X, y, label='True Data')
#plt.scatter(X, y_pred, label='GP Prediction')
#plt.legend()
#plt.show()

# 获取符号回归模型生成的公式
print("Best expression found: ", est._program)

# 逆归一化处理
def inverse_transform_formula(formula, scaler):
    min_val, max_val = scaler.data_min_[0], scaler.data_max_[0]
    range_val = max_val - min_val
    # 替换归一化变量
    formula_str = str(formula)
    transformed_formula = formula_str.replace("X0", f"({min_val} + {range_val} * X0)")
    return transformed_formula

# 输出逆归一化后的公式
inverse_formula = inverse_transform_formula(est._program, scaler)
print("Inverse transformed formula: ", inverse_formula)
