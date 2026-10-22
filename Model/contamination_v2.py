"""Repair the controlled contamination comparison using clean fold k and tau."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from Model.models_v2 import KnnNormalDistanceDetector, select_validation_threshold
from Model.experiment_v2 import compute_metrics

def main():
    folds=json.loads(Path('Dataset/v2/splits.json').read_text())['outer_folds']
    oof=pd.read_csv('Model/outputs/v2/oof_predictions.csv')
    ids=oof.instance_id.tolist(); idx={x:i for i,x in enumerate(ids)}; y=oof.true_label.to_numpy()
    log=np.load('Dataset/v2/embeddings/r2_minilm.npz'); patch=np.load('Dataset/v2/embeddings/r3_codebert.npz')
    li={str(x):i for i,x in enumerate(log['instance_ids'])}; pi={str(x):i for i,x in enumerate(patch['instance_ids'])}
    a=np.array([log['embeddings'][li[x]] for x in ids]); b=np.array([patch['embeddings'][pi[x]] for x in ids]); empty=np.array([patch['is_empty'][pi[x]] for x in ids])
    pars=pd.read_csv('Model/tables/v2/fold_parameters.csv'); rows=[]; records=[]; predictions=[]
    for seed in [42,123,456]:
        for p in [0,.05,.10]:
            scores=np.zeros(len(ids),dtype=np.float32); fixed=np.zeros(len(ids),int); recal=np.zeros(len(ids),int)
            for f in folds:
                fi=f['fold']; par=pars[(pars.fold==fi)&(pars.model=='knn_R3_T2')].iloc[0]
                k=int(par.chosen_param.split('=')[1]); clean_tau=float(par.tau)
                tr=f['train_instance_ids']; normals=f['train_normal_instance_ids']; failures=[x for x in tr if y[idx[x]]==1]
                n=len(normals); m=int(np.floor(n*p)); rng=np.random.RandomState(seed+1000*fi)
                normal_order=rng.permutation(normals).tolist(); failure_order=rng.permutation(failures).tolist()
                refs=normal_order[:n-m]+failure_order[:m]; ri=[idx[x] for x in refs]
                vi=[idx[x] for x in f['val_instance_ids']]; ti=[idx[x] for x in f['test_instance_ids']]
                model=KnnNormalDistanceDetector(k=k).fit_embeddings(a[ri],b[ri],empty[ri])
                v=model.score_dual(a[vi],b[vi],empty[vi]); tau,_=select_validation_threshold(y[vi],v)
                s=model.score_dual(a[ti],b[ti],empty[ti]); scores[ti]=s
                fixed[ti]=s>=clean_tau; recal[ti]=s>=tau
                records.append(dict(seed=seed,p_contamination=p,fold=fi,k=k,clean_tau=clean_tau,recalibrated_tau=tau,
                    fitting_ids=refs,injected_ids=failure_order[:m],actual_fraction=m/n,n_replaced=m,n_reference=n))
                for j,ix in enumerate(ti):
                    predictions.append(dict(seed=seed,p_contamination=p,fold=fi,instance_id=ids[ix],label=int(y[ix]),score=float(s[j]),
                        clean_tau=clean_tau,recalibrated_tau=tau,fixed_pred=int(fixed[ix]),recalibrated_pred=int(recal[ix])))
            if p==0:
                assert np.allclose(scores,oof.knn_R3_T2_seed_42_score,atol=1e-7)
                assert np.array_equal(fixed,oof.knn_R3_T2_seed_42_pred)
            for name,pred in [('clean_tau',fixed),('recalibrated_tau',recal)]:
                rows.append(dict(seed=seed,p_contamination=p,threshold_mode=name,**compute_metrics(y,scores,pred)))
    pd.DataFrame(rows).to_csv('Model/tables/v2/ablation_contamination.csv',index=False)
    pd.DataFrame(predictions).to_csv('Model/outputs/v2/contamination_predictions.csv',index=False)
    Path('Model/tables/v2/contamination_fitting_ids.json').write_text(json.dumps(records,indent=2)+'\n')
    print(pd.DataFrame(rows).groupby(['p_contamination','threshold_mode']).balanced_accuracy.mean().to_string())

if __name__=='__main__': main()
