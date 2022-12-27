# HSE_2022
The program performs prediction using two models: KNeighborsRegressor and KNeighborsClassifier. To start, you need to pass the following parameters: the name of the model and the parameters on which the prediction should be based (age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal).

###### Example how to run 
`python entrypoint.py --prediction_model KNeighborsClassifier,KNeighborsRegressor,Random Forest --prediction_params 47,1,0,110,275,0,0,118,1,1,1,1,2`
