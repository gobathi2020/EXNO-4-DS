# EXNO:4-DS
## AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

## ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

### FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

### FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

## CODING AND OUTPUT:
       from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, Normalizer, RobustScaler
       import pandas as pd
       df1 = pd.read_csv("bmi.csv")
       df1 
<img width="1128" height="419" alt="image" src="https://github.com/user-attachments/assets/a96bcf66-69fa-4055-91d2-ba15efcc0318" />

       df2 = df1.copy()
       enc = StandardScaler()
       df2[['new_height', 'new_weight']] = enc.fit_transform(df2[['Height', 'Weight']])
       df2
<img width="972" height="413" alt="image" src="https://github.com/user-attachments/assets/e507e632-40a6-4113-b37f-31ba6cbc77b8" />

       df3 = df1.copy()
       enc = MinMaxScaler()
       df3[['new_height', 'new_weight']] = enc.fit_transform(df3[['Height', 'Weight']])
       df3
<img width="978" height="415" alt="image" src="https://github.com/user-attachments/assets/b05a5706-7122-41fd-a2ed-04308a5cdd17" />

       df4 = df1.copy()
       enc = MaxAbsScaler()
       df4[['new_height', 'new_weight']] = enc.fit_transform(df4[['Height', 'Weight']])
       df4
<img width="1108" height="418" alt="image" src="https://github.com/user-attachments/assets/ffec9606-f161-4e51-94ba-d6211650207a" />

       df5 = df1.copy()
       enc = Normalizer()
       df5[['new_height', 'new_weight']] = enc.fit_transform(df5[['Height', 'Weight']])
       df5

<img width="998" height="416" alt="image" src="https://github.com/user-attachments/assets/a2e1c118-0a7e-4f9d-9003-0921a9ba4db2" />

        df6 = df1.copy()
        enc = RobustScaler()
        df6[['new_height', 'new_weight']] = enc.fit_transform(df6[['Height', 'Weight']])
        df6

<img width="1010" height="413" alt="image" src="https://github.com/user-attachments/assets/1aac53d1-7d94-44de-be7d-f947bd387f4f" />

       df=pd.read_csv("income(1) (1).csv")
       df

<img width="1440" height="418" alt="image" src="https://github.com/user-attachments/assets/9935ee1f-2e5d-49d6-b757-192e577be999" />

       from sklearn.preprocessing import LabelEncoder
       df_encoded=df.copy()
       le=LabelEncoder()
       for col in df_encoded.select_dtypes(include="object").columns:
              df_encoded[col] = le.fit_transform(df_encoded[col])
              x = df_encoded.drop("SalStat", axis=1)
              y = df_encoded["SalStat"]
       x

<img width="1235" height="422" alt="image" src="https://github.com/user-attachments/assets/a8379d56-de69-48ec-bd14-7a92961796d6" />

       from sklearn.feature_selection import SelectKBest, chi2
       chi2_selector=SelectKBest(chi2,k=5)
       chi2_selector.fit(x,y)
       selected_features_chi2=x.columns[chi2_selector.get_support()]
       print("Selected features (Chi-Square):",list(selected_features_chi2))
       mi_scores=pd.Series(chi2_selector.scores_,index=x.columns)
       print(mi_scores.sort_values(ascending=False))

<img width="1047" height="260" alt="image" src="https://github.com/user-attachments/assets/7083b8ba-f12c-413e-9d1e-a8c88d392748" />

       from sklearn.feature_selection import f_classif
       anova_selector=SelectKBest(f_classif,k=5)
       anova_selector.fit(x,y)
       selected_features_anova=x.columns[anova_selector.get_support()]
       print("Selected features (ANOVA F-test):",list(selected_features_anova))
       mi_scores=pd.Series(anova_selector.scores_,index=x.columns)
       print(mi_scores.sort_values(ascending=False))

<img width="1033" height="254" alt="image" src="https://github.com/user-attachments/assets/50857e5c-9688-4c6f-8bc8-3da06f010a3b" />

       from sklearn.feature_selection import f_classif
       anova_selector=SelectKBest(f_classif,k=5)
       anova_selector.fit(x,y)
       selected_features_anova=x.columns[anova_selector.get_support()]
       print("Selected features (ANOVA F-test):",list(selected_features_anova))
       mi_scores=pd.Series(anova_selector.scores_,index=x.columns)
       print(mi_scores.sort_values(ascending=False))

<img width="1042" height="251" alt="image" src="https://github.com/user-attachments/assets/fdb591ff-4904-4e93-a3cc-91e6d05f2c95" />


       from sklearn.linear_model import LogisticRegression
       from sklearn.feature_selection import RFE 
       model = LogisticRegression(max_iter=100)
       rfe = RFE(model, n_features_to_select=5)
       rfe.fit(x,y)
       selected_features_rfe=x.columns[rfe.support_]
       print("Selected features (RFE):", list(selected_features_rfe))

<img width="1277" height="685" alt="image" src="https://github.com/user-attachments/assets/9f09370d-4321-4ac0-a4c6-daa4e0959dff" />

       from sklearn.linear_model import LogisticRegression
       from sklearn.feature_selection import SequentialFeatureSelector
       model = LogisticRegression(max_iter=100)
       rfe = SequentialFeatureSelector(model, n_features_to_select=5)
       rfe.fit(x,y)
       selected_features_rfe=x.columns[rfe.support_]
       print("Selected features (SF):", list(selected_features_rfe))

<img width="1200" height="584" alt="image" src="https://github.com/user-attachments/assets/607272df-a323-47a9-ae0d-240708c9071b" />

       from sklearn.ensemble import RandomForestClassifier
       rf=RandomForestClassifier()
       rf.fit(x,y)
       importances=pd.Series(rf.feature_importances_,index=x.columns)
       selected_features_rf=importances.sort_values(ascending=False)
       print(importances)
       print("Selected features (RandomForestClassifier):",list(selected_features_rf))

<img width="1189" height="255" alt="image" src="https://github.com/user-attachments/assets/b8013295-4d1c-4647-bfaf-ab86c2e11055" />

       from sklearn.linear_model import LassoCV
       import numpy as np
       lasso=LassoCV(cv=5).fit(x,y)
       importance=np.abs(lasso.coef_)
       selected_features_lasso=x.columns[importance>0]
       print("Selected features (lasso):",list(selected_features_lasso))
<img width="1356" height="24" alt="image" src="https://github.com/user-attachments/assets/1e6e2d24-ff5b-404d-9789-ae1384457434" />

   import pandas as pd 
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import LabelEncoder,StandardScaler 
   from sklearn.neighbors import KNeighborsClassifier
   from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
   df=pd.read_csv("income(1) (1).csv")
   le=LabelEncoder()
   df_encoded=df.copy()

   for col in df_encoded.select_dtypes(include="object").columns:
    df_encoded[col]=le.fit_transform(df_encoded[col])

   x=df_encoded.drop("SalStat",axis=1)
   y=df_encoded["SalStat"]

   x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
   scaler=StandardScaler()
   x_train=scaler.fit_transform(x_train)
   x_test=scaler.transform(x_test)

   knn=KNeighborsClassifier(n_neighbors=3)
   knn.fit(x_train,y_train)
   y_pred=knn.predict(x_test)

   print("Accuracy:",accuracy_score(y_test,y_pred))
   print("\nConfusion Matrix:\n",confusion_matrix(y_test,y_pred))
   print("\nClassification Report:\n",classification_report(y_test,y_pred))

<img width="1050" height="276" alt="image" src="https://github.com/user-attachments/assets/e91d3a1e-0d3f-4954-a398-0ac195377fc9" />

## RESULT:
Thus, Feature selection and Feature scaling has been performed on the given dataset.
