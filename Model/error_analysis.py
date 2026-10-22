"""Evidence-backed case selection, feature attribution and illustrative impact."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from Model.models import AutoencoderDetector

OUT=Path('Model/tables')

def main():
    train,val,test=[pd.read_csv(f'Dataset/{s}.csv') for s in ['train','val','test']]
    cols=list(train.columns[5:])
    model=AutoencoderDetector(24,seed=42).fit(train.loc[train.label==0,cols].to_numpy())
    tau=model.tune_threshold(val[cols].to_numpy(),val.label.to_numpy())
    test['score']=model.score_samples(test[cols].to_numpy()); test['pred']=(test.score>=tau).astype(int)
    test['distance_to_tau']=test.score-tau
    fp=test[(test.label==0)&(test.pred==1)]
    fn=test[(test.label==1)&(test.pred==0)]
    if len(fp)<1 or len(fn)<2: raise ValueError('Required distinct FP/FN examples unavailable; inspect actual errors.')
    cases=pd.DataFrame([fp.sort_values('score',ascending=False).iloc[0],fn.sort_values('score',ascending=False).iloc[0],fn.sort_values('score').iloc[0]])
    cases.insert(0,'Type',['FP','FN boundary','FN low score'])
    cases['threshold']=tau
    cases.to_csv(OUT/'failure_cases.csv',index=False)
    # Exact recorded observations, not invented semantic diagnoses.
    excerpts=[]
    for row in cases.itertuples():
        data=json.loads(Path(f'Dataset/raw/sweagent_gpt4/{row.instance_id}.traj').read_bytes())
        excerpts.append({'instance_id':row.instance_id,'exit_status':row.exit_status,
            'final_action':data.get('trajectory',[{}])[-1].get('action',''),
            'final_observation':data.get('trajectory',[{}])[-1].get('observation','')})
    (OUT/'case_excerpts.json').write_text(json.dumps(excerpts,indent=2)+'\n')
    flagged=test[(test.label==1)&(test.pred==1)].iloc[0]
    x=torch.tensor(model.scaler.transform(flagged[cols].to_numpy(dtype=float).reshape(1,-1)),dtype=torch.float32)
    with torch.no_grad(): errors=((x-model.model(x))**2).numpy().ravel()
    attr=pd.DataFrame({'Instance_ID':flagged.instance_id,'Score':flagged.score,'Threshold':tau,
        'Feature':cols,'Reconstruction_MSE':errors,'Contribution_Percent':100*errors/errors.sum(),
        'Raw_Value':flagged[cols].to_numpy(dtype=float)})
    attr.sort_values('Contribution_Percent',ascending=False).to_csv(OUT/'feature_attribution_flagged_sample.csv',index=False)
    # One coherent hypothetical scenario using all five seeds' confusion matrices.
    impact=[]
    for seed in [42,123,456,789,2026]:
        pred=pd.read_csv(f'Model/outputs/test_predictions_seed_{seed}.csv')
        for tier in [1,2,3]:
            tn,fpn,fnn,tp=confusion_matrix(pred.label,pred[f'T{tier}_pred'],labels=[0,1]).ravel()
            recall=tp/(tp+fnn); fpr=fpn/(fpn+tn)
            caught=400*recall; false_alerts=100*fpr; reviews=caught+false_alerts
            impact.append(dict(seed=seed,Tier=tier,TP=int(tp),TN=int(tn),FP=int(fpn),FN=int(fnn),
                Recall=recall,FPR=fpr,Monthly_Caught=caught,Monthly_Escaped=400-caught,
                Monthly_False_Alerts=false_alerts,Monthly_Reviews=reviews,Review_Hours=reviews*.5,
                Monthly_Cost=(400-caught)*1200+reviews*30))
    pd.DataFrame(impact).to_csv(OUT/'practical_impact.csv',index=False)
    print(cases[['Type','instance_id','num_steps','num_error_observations','patch_lines_added','score','threshold','distance_to_tau']].to_string(index=False))
    print(attr.sort_values('Contribution_Percent',ascending=False).head(4).to_string(index=False))
    print(pd.DataFrame(impact).groupby('Tier').mean().to_string())

if __name__=='__main__': main()
