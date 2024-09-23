import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from pickle import load



#plt.style.use("style.mplstyle")
#shap.initjs() #notebook环境下

#data = pd.read_csv('../data_ts_el.csv')

data = pd.read_csv('../data_descriptor_select177.csv')

'''
features = ['ave:covalent_radius_pyykko', 'ave:covalent_radius_pyykko_double',
       'ave:en_allen', 'ave:vdw_radius_mm3', 'sum:dipole_polarizability',
       'sum:fusion_enthalpy', 'sum:hhi_p', 'sum:hhi_r',
       'sum:heat_of_formation', 'sum:num_unfilled', 'sum:num_s_unfilled',
       'sum:specific_heat', 'var:bulk_modulus', 'var:hhi_p', 'var:num_valance',
       'var:vdw_radius_uff', 'QT', 'TT', 'TT_time']

features_name = ['ave:CRP', 'ave:CRP_D',
       'ave:EA', 'ave:Vdwr_m', 'sum:DP',
       'sum:FE', 'sum:HHI_P', 'sum:HHI_R',
       'sum:HF', 'sum:NU_N', 'sum:NU_S',
       'sum:SH', 'var:BM', 'var:HHI_P', 'var:NU_V',
       'var:Vdwr_u', 'QT', 'TT', 'TT_time']
'''


features = ['ave:covalent_radius_pyykko_double', 'ave:hhi_p', 'ave:num_s_unfilled',
       'sum:fusion_enthalpy', 'var:bulk_modulus', 'var:num_p_unfilled',
       'var:num_s_unfilled', 'var:num_s_valence', 'var:vdw_radius_uff', 'TT',
       'TT_time']

features_name = ['ave:CRP_D', 'ave:HHI_P', 'ave:NU_S',
       'sum:FE', 'var:BM', 'var:NU_P',
       'var:NU_S', 'var:NU_SV','var:Vdwr_u', 'TT', 'TT_time']


X = data[features]
y = data.iloc[:, 177]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


model = load(open('../model_pickle/GBR_ts_descriptor_selected.pkl', 'rb'))
#explainer
explainer = shap.KernelExplainer(model.predict, X)
#explainer = shap.Explainer(Model)
shap_values = explainer(X)


#summary_bar
shap.plots.bar(shap.Explanation(shap_values, feature_names=features_name), max_display=99, show=False)
plt.gcf()
plt.tight_layout()
plt.title("Tensile strength shap_values")
plt.savefig("../explanation/shappng/GBR_ts_desriptor_summary.png")
plt.show()

#summay_beeswarm
shap.plots.beeswarm(shap.Explanation(shap_values, feature_names=features_name), max_display=99, show=False)
plt.gcf()
plt.tight_layout()
plt.title("Tensile strength shap_values")
plt.savefig("../explanation/shappng/GBR_ts_dexcriptor_beeswarm.png")
plt.show()


plt.subplot(2,1,1)
plt.gcf()
shap.plots.bar(shap.Explanation(shap_values.max(0), feature_names=features), max_display=99, show=False)
plt.subplot(2,1,2)
shap.plots.beeswarm(
    shap.Explanation(shap_values.abs, feature_names=features), color="shap_blue", max_display=99, show=False)

ax = plt.gca()
masv = {}
for feature in ax.get_yticklabels():
    name = feature.get_text()
    print(name)
    col_ind = X.columns.get_loc(name)
    print(col_ind)
    mean_abs_cv = np.mean(np.abs(shap_values.values[:, col_ind]))
    masv[name] = mean_abs_cv
ax.scatter(
    masv.values(),
    [i for i in range(len(X.columns))],
    zorder=99,
    label="Mean Absolute SHAP Value",
    c="k",
    marker="|",
    linewidths=3,
    s=100,
)
ax.legend(frameon=True)

plt.tight_layout()
#plt.savefig("../explanation/bar_beeswarm.png")
plt.show()

#scatter
plt.subplot(331)

shap.plots.scatter(shap.Explanation(shap_values[:, 0],feature_names="ave:covalent_radius_pyykko"),show=False)
plt.gcf()
plt.tight_layout()
plt.savefig("../explanation/shappng/bulk_modulus_ts")
plt.show()

'''
#denpendence
shap.dependence_plot('Cr', shap_values, X, show=False, interaction_index='Ni')
plt.gcf()
plt.tight_layout()
#plt.savefig("../explanation/dependence_Si.png")
plt.show()
'''


# 设置字体和样式
plt.style.use('seaborn-whitegrid')  # 使用Seaborn的白色网格风格
plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体为Times New Roman
plt.rcParams['font.size'] = 14  # 设置全局字体大小

# 自定义颜色渐变方案（从蓝到红）
#custom_cmap = sns.color_palette("RdBu_r", as_cmap=True)  # 蓝到红渐变

# 绘制 SHAP beeswarm 图
shap_values = shap.Explanation(shap_values, feature_names=features_name)
shap.plots.beeswarm(
    shap_values,
    max_display=99,  # 应用自定义配色
    show=False
)

# 获取当前图形对象
fig = plt.gcf()

# 调整数据点为立体球形
for collection in plt.gca().collections:
    collection.set_edgecolor('k')  # 为每个点加上黑色边缘，增强立体感
    collection.set_linewidth(0.3)  # 调整边缘的线宽
    collection.set_antialiased(True)  # 平滑处理以增强效果


plt.tight_layout(pad=2.0)  # 增加pad值，确保标题显示完整
plt.subplots_adjust(top=0.9)  # 调整图表上边距

# 设置标题和标签字体大小
plt.title("Tensile Strength SHAP Values", fontsize=18, fontweight='bold')
plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=16)
plt.ylabel("Feature", fontsize=16)

# 保存为高分辨率图片
plt.savefig("../explanation/shappng/GBR_ts_descriptor_beeswarm.png", dpi=300, bbox_inches='tight')

# 显示图表
plt.show()
