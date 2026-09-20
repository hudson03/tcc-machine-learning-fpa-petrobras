import json, math, time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
source=json.loads((ROOT/"dados/ml_source.json").read_text(encoding="utf-8"))
headers=source["headers"]
idx={h:i for i,h in enumerate(headers)}

def val(row,name):
    v=row[idx[name]]
    return float(v) if isinstance(v,(int,float)) and math.isfinite(v) else 0.0

records=[]
for row in source["rows"]:
    records.append({
        "period":str(row[idx["Periodo"]]),"year":int(row[idx["Ano"]]),"quarter":int(row[idx["Trimestre"]]),
        "qid":int(row[idx["Quarter_ID"]]),"segment":str(row[idx["Segmento_Analitico"]]),"type":str(row[idx["Tipo_Segmento"]]),
        "cost":val(row,"Target_Custo"),"revenue":val(row,"Receita"),"expenses":val(row,"Despesas_Abs"),"ebit":val(row,"EBIT")
    })
panel={(r["segment"],r["qid"]):r for r in records}
segments=sorted({r["segment"] for r in records})
segment_features=[f"Segmento={s}" for s in segments[1:]]
numeric_names=["Custo_t","Custo_t_1","Custo_t_4","Receita_t","Receita_t_1","Receita_t_4","Despesas_t","Despesas_t_1","Despesas_t_4","EBIT_t","EBIT_t_1","EBIT_t_4","Trimestre_Sin","Trimestre_Cos"]
feature_names=numeric_names+segment_features

def make_x(origin):
    p1=panel[(origin["segment"],origin["qid"]-1)]
    p4=panel[(origin["segment"],origin["qid"]-4)]
    q=origin["quarter"]
    nums=[origin["cost"],p1["cost"],p4["cost"],origin["revenue"],p1["revenue"],p4["revenue"],origin["expenses"],p1["expenses"],p4["expenses"],origin["ebit"],p1["ebit"],p4["ebit"],math.sin(2*math.pi*q/4),math.cos(2*math.pi*q/4)]
    cats=[1.0 if origin["segment"]==s else 0.0 for s in segments[1:]]
    return np.asarray(nums+cats,dtype=float)

datasets={}
latest_qid=max(r["qid"] for r in records)
latest_origins=[]
for r in records:
    if (r["segment"],r["qid"]-1) in panel and (r["segment"],r["qid"]-4) in panel:
        if r["qid"]==latest_qid: latest_origins.append(r)

for h in range(1,5):
    ds=[]
    for origin in records:
        if (origin["segment"],origin["qid"]-1) not in panel or (origin["segment"],origin["qid"]-4) not in panel: continue
        target=panel.get((origin["segment"],origin["qid"]+h))
        if not target: continue
        seasonal=panel.get((origin["segment"],origin["qid"]+h-4))
        ds.append({"x":make_x(origin),"y":target["cost"],"origin_qid":origin["qid"],"origin_period":origin["period"],"target_qid":target["qid"],"target_period":target["period"],"segment":origin["segment"],"baseline_last":origin["cost"],"baseline_seasonal":seasonal["cost"] if seasonal else np.nan})
    datasets[h]=ds

def metrics(y,p):
    y=np.asarray(y,float);p=np.maximum(np.asarray(p,float),0)
    err=p-y
    denom=np.sum(np.abs(y))
    sst=np.sum((y-y.mean())**2)
    return {"N":int(len(y)),"WMAPE":float(np.sum(np.abs(err))/denom) if denom else None,"MAE":float(np.mean(np.abs(err))),"RMSE":float(np.sqrt(np.mean(err**2))),"R2":float(1-np.sum(err**2)/sst) if sst else None,"Bias":float(np.mean(err)),"Bias_Pct":float(np.sum(err)/denom) if denom else None}

class RidgeModel:
    def __init__(self,alpha): self.alpha=alpha
    def fit(self,X,y):
        self.mean=X.mean(axis=0);self.std=X.std(axis=0);self.std[self.std<1e-12]=1.0
        Z=(X-self.mean)/self.std;self.ymean=float(np.mean(y))
        self.beta=np.linalg.solve(Z.T@Z+self.alpha*np.eye(Z.shape[1]),Z.T@(y-self.ymean))
        imp=np.abs(self.beta);self.importance=imp/imp.sum() if imp.sum() else imp
        return self
    def predict(self,X): return np.maximum(0,self.ymean+((X-self.mean)/self.std)@self.beta)

class TreeModel:
    def __init__(self,max_depth=5,min_leaf=4,max_features=None,thresholds=8,rng=None):
        self.max_depth=max_depth;self.min_leaf=min_leaf;self.max_features=max_features;self.thresholds=thresholds;self.rng=rng or np.random.default_rng(0);self.gains=None
    def fit(self,X,y):
        self.p=X.shape[1];self.gains=np.zeros(self.p);self.root=self._build(X,y,0);return self
    def _build(self,X,y,depth):
        node={"value":float(np.mean(y)),"feature":None,"threshold":None,"left":None,"right":None}
        if depth>=self.max_depth or len(y)<2*self.min_leaf or np.var(y)<1e-12:return node
        features=np.arange(self.p)
        if self.max_features and self.max_features<self.p:features=self.rng.choice(features,self.max_features,replace=False)
        base=np.sum((y-y.mean())**2);best=None
        for j in features:
            x=X[:,j];uniq=np.unique(x)
            if len(uniq)<2:continue
            if len(uniq)>self.thresholds+1:
                qs=np.linspace(0.08,0.92,self.thresholds);ths=np.unique(np.quantile(x,qs))
            else:ths=(uniq[:-1]+uniq[1:])/2
            for t in ths:
                mask=x<=t;nl=int(mask.sum());nr=len(y)-nl
                if nl<self.min_leaf or nr<self.min_leaf:continue
                yl=y[mask];yr=y[~mask]
                loss=np.sum((yl-yl.mean())**2)+np.sum((yr-yr.mean())**2)
                gain=base-loss
                if best is None or gain>best[0]:best=(gain,j,float(t),mask)
        if best is None or best[0]<=1e-9:return node
        gain,j,t,mask=best;self.gains[j]+=gain
        node.update(feature=int(j),threshold=t,left=self._build(X[mask],y[mask],depth+1),right=self._build(X[~mask],y[~mask],depth+1))
        return node
    def _pred_one(self,x,node):
        while node["feature"] is not None:node=node["left"] if x[node["feature"]]<=node["threshold"] else node["right"]
        return node["value"]
    def predict(self,X):return np.asarray([self._pred_one(x,self.root) for x in X])

class RandomForestModel:
    def __init__(self,n_trees=35,max_depth=6,min_leaf=4,seed=0):self.n_trees=n_trees;self.max_depth=max_depth;self.min_leaf=min_leaf;self.seed=seed
    def fit(self,X,y):
        rng=np.random.default_rng(self.seed);self.trees=[];g=np.zeros(X.shape[1]);mf=max(1,int(math.sqrt(X.shape[1])))
        for i in range(self.n_trees):
            ids=rng.integers(0,len(y),len(y));tree=TreeModel(self.max_depth,self.min_leaf,mf,8,np.random.default_rng(self.seed+i+1)).fit(X[ids],y[ids]);self.trees.append(tree);g+=tree.gains
        self.importance=g/g.sum() if g.sum() else g;return self
    def predict(self,X):return np.maximum(0,np.mean([t.predict(X) for t in self.trees],axis=0))

class GradientBoostingModel:
    def __init__(self,n_estimators=60,learning_rate=.05,max_depth=2,min_leaf=5,seed=0):self.n_estimators=n_estimators;self.lr=learning_rate;self.max_depth=max_depth;self.min_leaf=min_leaf;self.seed=seed
    def fit(self,X,y):
        self.base=float(np.mean(y));pred=np.full(len(y),self.base);self.trees=[];g=np.zeros(X.shape[1])
        for i in range(self.n_estimators):
            residual=y-pred;tree=TreeModel(self.max_depth,self.min_leaf,None,8,np.random.default_rng(self.seed+i)).fit(X,residual);update=tree.predict(X);pred+=self.lr*update;self.trees.append(tree);g+=tree.gains
        self.importance=g/g.sum() if g.sum() else g;return self
    def predict(self,X):
        p=np.full(len(X),self.base)
        for t in self.trees:p+=self.lr*t.predict(X)
        return np.maximum(0,p)

ridge_grid=[{"alpha":a} for a in (.1,1.0,10.0,100.0)]
rf_grid=[{"n_trees":35,"max_depth":5,"min_leaf":4},{"n_trees":35,"max_depth":7,"min_leaf":3}]
gb_grid=[{"n_estimators":60,"learning_rate":.05,"max_depth":2,"min_leaf":5},{"n_estimators":80,"learning_rate":.03,"max_depth":3,"min_leaf":4}]

def build_model(name,param,seed):
    if name=="Ridge":return RidgeModel(**param)
    if name=="Random Forest":return RandomForestModel(seed=seed,**param)
    return GradientBoostingModel(seed=seed,**param)

def arrays(items):return np.stack([d["x"] for d in items]),np.asarray([d["y"] for d in items],float)

tuning=[];best_params={};all_predictions=[];all_metrics=[];selected={};final_models={};importance=[]
model_grids={"Ridge":ridge_grid,"Random Forest":rf_grid,"Gradient Boosting":gb_grid}
for h in range(1,5):
    ds=datasets[h]
    subtrain=[d for d in ds if d["target_qid"]<=44];internal=[d for d in ds if 45<=d["target_qid"]<=52]
    Xtr,ytr=arrays(subtrain);Xiv,yiv=arrays(internal)
    for name,grid in model_grids.items():
        scored=[]
        for k,param in enumerate(grid):
            model=build_model(name,param,10000+h*100+k).fit(Xtr,ytr);pred=model.predict(Xiv);m=metrics(yiv,pred);scored.append((m["WMAPE"],param));tuning.append({"Horizonte":f"H{h}","Modelo":name,"Parametros":json.dumps(param,ensure_ascii=False),"WMAPE_Interno":m["WMAPE"],"N_Interno":m["N"]})
        best_params[(h,name)]=min(scored,key=lambda z:z[0])[1]
    print(f"H{h}: parâmetros internos selecionados",flush=True)

    # Expanding-window validation and test for all candidates.
    for split,qstart,qend in [("Validação",53,68),("Teste",69,82)]:
        pred_by_model={n:[] for n in ["Ridge","Random Forest","Gradient Boosting","Baseline Último","Baseline Sazonal"]};actual=[];meta=[]
        for q in range(qstart,qend+1):
            train=[d for d in ds if d["target_qid"]<q];testq=[d for d in ds if d["target_qid"]==q]
            if not testq:continue
            Xtr,ytr=arrays(train);Xq,yq=arrays(testq);actual.extend(yq.tolist());meta.extend(testq)
            for mi,name in enumerate(["Ridge","Random Forest","Gradient Boosting"]):
                model=build_model(name,best_params[(h,name)],20000+h*1000+q*10+mi).fit(Xtr,ytr);pred_by_model[name].extend(model.predict(Xq).tolist())
            pred_by_model["Baseline Último"].extend([d["baseline_last"] for d in testq]);pred_by_model["Baseline Sazonal"].extend([d["baseline_seasonal"] for d in testq])
        for name,preds in pred_by_model.items():
            m=metrics(actual,preds);all_metrics.append({"Horizonte":f"H{h}","Bloco":split,"Modelo":name,**m})
            for d,a,p in zip(meta,actual,preds):all_predictions.append({"Horizonte":f"H{h}","Bloco":split,"Modelo":name,"Origem_Periodo":d["origin_period"],"Target_Periodo":d["target_period"],"Target_Quarter_ID":d["target_qid"],"Segmento":d["segment"],"Actual":a,"Forecast":max(0,p),"Erro":max(0,p)-a,"Erro_Abs":abs(max(0,p)-a)})

    val_metrics=[m for m in all_metrics if m["Horizonte"]==f"H{h}" and m["Bloco"]=="Validação"]
    best_overall=min(val_metrics,key=lambda m:m["WMAPE"])["Modelo"]
    best_ml=min([m for m in val_metrics if m["Modelo"] in model_grids],key=lambda m:m["WMAPE"])["Modelo"]
    selected[h]={"best_overall":best_overall,"best_ml":best_ml,"best_ml_params":best_params[(h,best_ml)]}

    # Final model for driver importance and the next forecast origin.
    train_all=ds;Xall,yall=arrays(train_all)
    ml_model=build_model(best_ml,best_params[(h,best_ml)],50000+h).fit(Xall,yall);final_models[h]=ml_model
    for f,imp in sorted(zip(feature_names,ml_model.importance),key=lambda z:z[1],reverse=True):importance.append({"Horizonte":f"H{h}","Modelo":best_ml,"Driver":f,"Importancia":float(imp)})
    print(f"H{h}: backtest concluído; selecionado {best_overall} (melhor ML: {best_ml})",flush=True)

# Compact selected-model backtest.
backtest=[]
for h in range(1,5):
    chosen=selected[h]["best_overall"]
    chosen_rows=[r for r in all_predictions if r["Horizonte"]==f"H{h}" and r["Modelo"]==chosen]
    last_lookup={(r["Bloco"],r["Target_Quarter_ID"],r["Segmento"]):r for r in all_predictions if r["Horizonte"]==f"H{h}" and r["Modelo"]=="Baseline Último"}
    seas_lookup={(r["Bloco"],r["Target_Quarter_ID"],r["Segmento"]):r for r in all_predictions if r["Horizonte"]==f"H{h}" and r["Modelo"]=="Baseline Sazonal"}
    for r in chosen_rows:
        key=(r["Bloco"],r["Target_Quarter_ID"],r["Segmento"]);backtest.append({**r,"Modelo_Selecionado":chosen,"Baseline_Ultimo":last_lookup[key]["Forecast"],"Baseline_Sazonal":seas_lookup[key]["Forecast"]})

# Future rolling forecast at latest origin.
forecast=[]
for origin in sorted(latest_origins,key=lambda r:r["segment"]):
    row={"Segmento":origin["segment"],"Origem_Periodo":origin["period"],"Custo_Actual":origin["cost"]}
    x=make_x(origin).reshape(1,-1)
    total=0.0
    for h in range(1,5):
        chosen=selected[h]["best_overall"]
        if chosen=="Baseline Último":pred=origin["cost"]
        elif chosen=="Baseline Sazonal":pred=panel[(origin["segment"],origin["qid"]+h-4)]["cost"]
        else:
            model=build_model(chosen,best_params[(h,chosen)],60000+h).fit(*arrays(datasets[h]));pred=float(model.predict(x)[0])
        row[f"H{h}"]=max(0,float(pred));row[f"Modelo_H{h}"]=chosen;total+=row[f"H{h}"]
    row["Total_12M"]=total;forecast.append(row)

# Selected summary with test comparison.
selection=[]
for h in range(1,5):
    for kind,model_name in [("Modelo operacional",selected[h]["best_overall"]),("Melhor ML",selected[h]["best_ml"])]:
        vm=next(m for m in all_metrics if m["Horizonte"]==f"H{h}" and m["Bloco"]=="Validação" and m["Modelo"]==model_name)
        tm=next(m for m in all_metrics if m["Horizonte"]==f"H{h}" and m["Bloco"]=="Teste" and m["Modelo"]==model_name)
        selection.append({"Horizonte":f"H{h}","Tipo":kind,"Modelo":model_name,"Parametros":json.dumps(best_params.get((h,model_name),{}),ensure_ascii=False),"WMAPE_Validacao":vm["WMAPE"],"WMAPE_Teste":tm["WMAPE"],"MAE_Teste":tm["MAE"],"RMSE_Teste":tm["RMSE"],"R2_Teste":tm["R2"],"Bias_Pct_Teste":tm["Bias_Pct"]})

result={"metadata":{"generated_at":"2026-08-30","method":"Direct multi-horizon expanding-window backtest","features":feature_names,"segments":segments,"latest_qid":latest_qid},"tuning":tuning,"best_params":[{"Horizonte":f"H{h}","Modelo":n,"Parametros":p} for (h,n),p in best_params.items()],"metrics":all_metrics,"selected":selection,"backtest":backtest,"forecast":forecast,"importance":importance}
(ROOT/"reproducao").mkdir(parents=True,exist_ok=True)
(ROOT/"reproducao/model_results.json").write_text(json.dumps(result,ensure_ascii=False),encoding="utf-8")
print(json.dumps({"selected":selected,"forecast":forecast,"test_metrics":[m for m in all_metrics if m["Bloco"]=="Teste"]},ensure_ascii=False),flush=True)
