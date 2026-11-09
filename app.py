import os,pandas as pd,streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
d=pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"credit_data.csv")); X=d.drop("CreditRisk",axis=1); y=d.CreditRisk
xt,xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
M={"Logistic Regression":Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler()),("m",LogisticRegression(max_iter=2000))]),"Decision Tree":Pipeline([("i",SimpleImputer(strategy="median")),("m",DecisionTreeClassifier(max_depth=6,random_state=42))]),"Random Forest":Pipeline([("i",SimpleImputer(strategy="median")),("m",RandomForestClassifier(n_estimators=200,max_depth=8,random_state=42))])}
rows=[]
for n,m in M.items():
 m.fit(xt,yt); p=m.predict(xv); q=m.predict_proba(xv)[:,1]; rows.append([n,accuracy_score(yv,p),precision_score(yv,p,zero_division=0),recall_score(yv,p,zero_division=0),f1_score(yv,p,zero_division=0),roc_auc_score(yv,q)])
r=pd.DataFrame(rows,columns=["Model","Accuracy","Precision","Recall","F1","ROC-AUC"]).sort_values("ROC-AUC",ascending=False); best=M[r.iloc[0].Model]
st.title("💳 Credit Scoring Model"); st.write("Educational ML demonstration.")
a=st.number_input("Age",18,100,30); inc=st.number_input("Annual Income",0.0,1000000.0,60000.0); loan=st.number_input("Loan Amount",0.0,1000000.0,12000.0); debt=st.number_input("Current Debt",0.0,1000000.0,5000.0); cs=st.number_input("Credit Score",300,850,720); ph=st.number_input("Payment History (0-100)",0.0,100.0,95.0); ey=st.number_input("Employment Years",0.0,60.0,6.0)
if st.button("Predict Creditworthiness"):
 x=pd.DataFrame([{"Age":a,"Income":inc,"LoanAmount":loan,"Debt":debt,"CreditScore":cs,"PaymentHistory":ph,"EmploymentYears":ey}]); p=int(best.predict(x)[0]); q=best.predict_proba(x)[0,1]
 (st.success if p else st.error)("Prediction: GOOD CREDIT" if p else "Prediction: POOR CREDIT"); st.metric("Probability of Good Credit",f"{q:.2%}")
st.subheader("Model Comparison"); st.dataframe(r,use_container_width=True); st.info(f"Best model by ROC-AUC: {r.iloc[0].Model}. Synthetic data; educational use only.")
