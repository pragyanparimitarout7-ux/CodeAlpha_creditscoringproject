import os, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,classification_report,ConfusionMatrixDisplay,RocCurveDisplay

BASE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(BASE,"credit_data.csv"); OUT=os.path.join(BASE,"results"); os.makedirs(OUT,exist_ok=True)
df=pd.read_csv(DATA); X=df.drop("CreditRisk",axis=1); y=df["CreditRisk"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
models={
"Logistic Regression":Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",LogisticRegression(max_iter=2000))]),
"Decision Tree":Pipeline([("imputer",SimpleImputer(strategy="median")),("model",DecisionTreeClassifier(max_depth=6,random_state=42))]),
"Random Forest":Pipeline([("imputer",SimpleImputer(strategy="median")),("model",RandomForestClassifier(n_estimators=200,max_depth=8,random_state=42))])}
rows=[]
for name,m in models.items():
    m.fit(Xtr,ytr); pred=m.predict(Xte); pr=m.predict_proba(Xte)[:,1]
    rows.append([name,accuracy_score(yte,pred),precision_score(yte,pred,zero_division=0),recall_score(yte,pred,zero_division=0),f1_score(yte,pred,zero_division=0),roc_auc_score(yte,pr)])
res=pd.DataFrame(rows,columns=["Model","Accuracy","Precision","Recall","F1_Score","ROC_AUC"]).sort_values("ROC_AUC",ascending=False)
print(res.to_string(index=False)); res.to_csv(os.path.join(OUT,"model_comparison.csv"),index=False)
best=models[res.iloc[0].Model]; pred=best.predict(Xte); pr=best.predict_proba(Xte)[:,1]
print("\nBEST MODEL:",res.iloc[0].Model); print(classification_report(yte,pred,target_names=["Poor Credit","Good Credit"],zero_division=0))
fig,ax=plt.subplots(figsize=(6,5)); ConfusionMatrixDisplay.from_predictions(yte,pred,display_labels=["Poor Credit","Good Credit"],ax=ax); plt.tight_layout(); plt.savefig(os.path.join(OUT,"confusion_matrix.png")); plt.close()
res.set_index("Model").plot(kind="bar",figsize=(10,6)); plt.ylim(0,1.05); plt.tight_layout(); plt.savefig(os.path.join(OUT,"model_comparison.png")); plt.close()
fig,ax=plt.subplots(figsize=(7,5)); RocCurveDisplay.from_predictions(yte,pr,ax=ax); plt.tight_layout(); plt.savefig(os.path.join(OUT,"roc_curve.png")); plt.close()
rf=models["Random Forest"]; rf.fit(Xtr,ytr); imp=pd.Series(rf.named_steps["model"].feature_importances_,index=X.columns).sort_values(); imp.plot(kind="barh",figsize=(8,5)); plt.tight_layout(); plt.savefig(os.path.join(OUT,"feature_importance.png")); plt.close()

def interactive():
    print("\n=== CREDIT SCORE PREDICTION ===")
    try:
        a=pd.DataFrame([{"Age":float(input("Age: ")),"Income":float(input("Annual Income: ")),"LoanAmount":float(input("Loan Amount: ")),"Debt":float(input("Current Debt: ")),"CreditScore":float(input("Credit Score (300-850): ")),"PaymentHistory":float(input("Payment History (0-100): ")),"EmploymentYears":float(input("Employment Years: "))}])
        p=int(best.predict(a)[0]); q=float(best.predict_proba(a)[0,1])
        print("\nPrediction:", "GOOD CREDIT" if p else "POOR CREDIT"); print(f"Probability of Good Credit: {q:.2%}")
    except ValueError: print("Please enter numbers only.")
if __name__=="__main__": interactive()
