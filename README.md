# DE-FIS-model-MOP
In this study, the chemical element compositions of low-alloy steel data are transformed into physicochemical descriptors to optimize the machine learning hyperparameters by differential evolutionary algorithm. A two-stage feature selection algorithm (DE_FIS) based on differential evolutionary algorithm and model importance ranking is proposed. The tensile strength and ductility prediction models constructed based on the filtered descriptors. In addition, we conducted an interpretability analysis of the model using SHAP values to identify the key features that affect the mechanical properties of low-alloy steel. Through symbolic regression (SR) based on genetic planning, a quantitative relationship formula between the key features and the target performance was established. Finally, a meta-heuristic search was performed using a multi-objective optimization algorithm to provide a candidate solution set for the composition design of high-strength and high-ductility low-alloy steels.

1. Data preprocessing and data transformation  
Code: pre_data.py  
Impute and Convert chemical elements in raw data to descriptors  

2. Feature selection(DE-FIS)  
Input: Combine transformed descriptors with process parameters  
code :DE-FIS.py  
Output: selected features  

3. Machine learning Model Build and Hyperparameter optimization  
code:model.py  
dump model(tensile strength and elongation)

4. Predict Online Web  
   code: predict.py  
   
5. Model Interpretability Analysis  
   (1) SHAP --summary.plot&beeswarm.plot  
     code:shap_explain.py  
   (2)SRGP --Mathematical formulas for key features and performance  
     code:SRGP.py
   
6. Multi-objective optimization  
   code:MOP.py  
   Output--Pareto Font  
