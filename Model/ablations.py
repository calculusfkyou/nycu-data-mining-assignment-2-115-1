"""Controlled weight-contamination, masked-feature, and threshold ablations.

Only training rows may enter gradient updates. All runs keep 32 training rows,
24 inputs, the 24-16-6-16-24 architecture, optimizer and seeds fixed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, fbeta_score
from sklearn.preprocessing import StandardScaler
from Model.models import AutoencoderDetector

SEEDS = [42, 123, 456, 789, 2026]
OUT = Path('Model/tables')
META = ['instance_id', 'repo', 'label', 'is_resolved', 'exit_status']

def main():
    train, val, test = [pd.read_csv(f'Dataset/{s}.csv') for s in ['train', 'val', 'test']]
    cols = [c for c in train if c not in META]
    clean = train[train.label == 0][cols].to_numpy()
    failures = train[train.label == 1][cols].to_numpy()
    xv, xt = val[cols].to_numpy(), test[cols].to_numpy()
    yv, yt = val.label.to_numpy(), test.label.to_numpy()
    records = []
    fitting_ids = {}
    for seed in SEEDS:
        order = np.random.default_rng(seed).permutation(len(failures))
        for k in [0, 1, 2, 3]:
            x = clean.copy()
            if k: x[-k:] = failures[order[:k]]
            ids = train.loc[train.label == 0, 'instance_id'].tolist()
            if k:
                ids[-k:] = train.loc[train.label == 1, 'instance_id'].iloc[order[:k]].tolist()
            fitting_ids[f'{seed}:{k}'] = ids
            # Freeze clean scaler to isolate contamination of weight fitting only.
            model = AutoencoderDetector(24, seed=seed).fit(x, scaler=StandardScaler().fit(clean))
            model.tune_threshold(xv, yv)
            scores = model.score_samples(xt)
            records.append(dict(experiment='contamination', setting=f'{100*k/len(clean):.3f}%',
                                count=k, seed=seed, PR_AUC=average_precision_score(yt,scores),
                                ROC_AUC=roc_auc_score(yt,scores), F2=fbeta_score(yt,scores>=model.threshold,beta=2)))
        for name, active in [('Full', list(range(24))), ('Process', list(range(16))), ('Patch', list(range(16,24)))]:
            mask = np.zeros(24); mask[active] = 1
            # Center all inputs using clean normal statistics before zero masking.
            scaler = StandardScaler().fit(clean)
            a,b,c = [scaler.transform(z)*mask for z in [clean,xv,xt]]
            # Identity scaler prevents reintroducing inactive columns.
            identity = StandardScaler(with_mean=False, with_std=False).fit(a)
            model = AutoencoderDetector(24, seed=seed).fit(a, scaler=identity)
            model.tune_threshold(b,yv)
            score = model.score_samples(c)
            records.append(dict(experiment='features',setting=name,count=len(active),seed=seed,
                                PR_AUC=average_precision_score(yt,score),ROC_AUC=roc_auc_score(yt,score),
                                F2=fbeta_score(yt,score>=model.threshold,beta=2)))
        # Reuse the full model's scores; only the threshold rule changes.
        v = pd.read_csv(f'Model/outputs/val_predictions_seed_{seed}.csv')
        t = pd.read_csv(f'Model/outputs/test_predictions_seed_{seed}.csv')
        for rule,beta in [('F1',1),('F2',2),('Normal95',None)]:
            if beta:
                candidates=np.percentile(v.T2_score,np.linspace(5,95,91))
                tau=max(candidates,key=lambda z: fbeta_score(yv,v.T2_score>=z,beta=beta,zero_division=0))
            else: tau=np.percentile(v.loc[v.label==0,'T2_score'],95)
            pred=t.T2_score>=tau
            from sklearn.metrics import confusion_matrix
            tn,fp,fn,tp=confusion_matrix(yt,pred,labels=[0,1]).ravel()
            records.append(dict(experiment='threshold',setting=rule,count=24,seed=seed,
                PR_AUC=average_precision_score(yt,t.T2_score),ROC_AUC=roc_auc_score(yt,t.T2_score),
                F2=fbeta_score(yt,pred,beta=2),FN=fn,FP=fp,Threshold=tau))
    runs=pd.DataFrame(records); runs.to_csv(OUT/'ablation_runs.csv',index=False)
    import json
    (OUT/'ablation_fitting_ids.json').write_text(json.dumps(fitting_ids, indent=2) + '\n')
    for ex,file,index in [('contamination','ablation_contamination.csv','Contamination_Ratio'),
                          ('features','ablation_features.csv','Feature_Set'),
                          ('threshold','ablation_threshold.csv','Threshold_Rule')]:
        sub=runs[runs.experiment==ex]; rows=[]
        for setting in sub.setting.unique():
            d=sub[sub.setting==setting]; row={index:setting,'Count':int(d['count'].iloc[0])}
            for metric in ['PR_AUC','ROC_AUC','F2','FN','FP']:
                if metric in d and d[metric].notna().any():
                    row[metric+'_mean']=d[metric].mean(); row[metric+'_std']=d[metric].std(ddof=1)
            rows.append(row)
        result=pd.DataFrame(rows)
        for metric in ['PR_AUC','F2']:
            result['Delta_'+metric]=result[metric+'_mean']-result[metric+'_mean'].iloc[0]
        result.to_csv(OUT/file,index=False); print(result.to_string(index=False))

if __name__=='__main__': main()
