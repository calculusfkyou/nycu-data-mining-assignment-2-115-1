"""Independent artifact/data checks. Exit nonzero on inconsistency.

This checks local evidence; it cannot certify a grade, E3 receipt, public GitHub
availability, member contributions, or an eight-minute live presentation.
"""
import hashlib
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from pypdf import PdfReader
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import (average_precision_score,roc_auc_score,f1_score,fbeta_score,precision_score,recall_score,confusion_matrix)
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from Dataset.download_and_extract import extract_features_from_trajectory
from Model.models import AutoencoderDetector

checks=0
def check(value,message):
    global checks
    if not bool(value): raise AssertionError(message)
    checks+=1

def main():
    check(Path.cwd()==ROOT,'Run from repository root')
    team=json.loads(Path('Paper/team.json').read_text(encoding='utf-8-sig'))
    frames={s:pd.read_csv(f'Dataset/{s}.csv') for s in ['train','val','test']}
    full=pd.read_csv('Dataset/sweagent_gpt4_features.csv')
    cols=list(full.columns[5:]);check(len(cols)==24,'24 feature columns')
    check('exit_cost_forced' not in cols,'Termination status excluded from patch features')
    check(full.instance_id.is_unique,'Unique run IDs')
    splits=json.loads(Path('Dataset/split_ids.json').read_text())
    for s,df in frames.items():
        check(df.instance_id.tolist()==splits[s],f'{s} frozen IDs')
        check(np.isfinite(df[cols].to_numpy()).all(),f'{s} numeric finite inputs')
    sets={s:set(d.instance_id) for s,d in frames.items()}
    for a,b in [('train','val'),('train','test'),('val','test')]:check(not sets[a]&sets[b],f'{a}/{b} disjoint')
    check(set.union(*sets.values())==set(full.instance_id),'Splits cover full dataset')
    check([(len(df),(df.label==0).sum(),(df.label==1).sum()) for df in frames.values()]==[(82,32,50),(40,10,30),(162,12,150)],'Split class counts')
    truth=json.loads(Path('Dataset/evaluation_results.json').read_text())
    check(set(full.instance_id)==set(truth['generated']),'All and only official generated trajectories')
    for row in full.itertuples():check(row.label==int(row.instance_id not in truth['resolved']),f'Official label {row.instance_id}')
    manifest=json.loads(Path('Dataset/provenance.json').read_text())
    check(len(manifest['trajectories'])==284,'Raw provenance coverage')
    for item in manifest['trajectories']:
        raw=Path('Dataset/raw/sweagent_gpt4')/(item['instance_id']+'.traj')
        check(hashlib.sha256(raw.read_bytes()).hexdigest()==item['sha256'],'Raw hash '+item['instance_id'])
        row=full.loc[full.instance_id==item['instance_id']].iloc[0]
        rebuilt=extract_features_from_trajectory(json.loads(raw.read_bytes()), item['instance_id'], bool(row.is_resolved))
        check(np.allclose([rebuilt[c] for c in cols],row[cols].to_numpy(dtype=float)), 'Raw-to-feature extraction '+item['instance_id'])
    # Clearly synthetic unit fixture: frequency != consecutive run length.
    fixture={'trajectory':[{'action':a,'observation':str(i)} for i,a in enumerate(['search','edit','search','search','edit','search'])]}
    check(extract_features_from_trajectory(fixture,'fixture__test',False)['max_repeated_tool_streak']==2,'Consecutive streak, not tool frequency')
    train=frames['train'];val=frames['val'];test=frames['test']; yt=test.label.to_numpy()
    sep=pd.read_csv('EDA/tables/feature_separability.csv').set_index('Feature')
    scaled=StandardScaler().fit_transform(train[cols])
    for i,col in enumerate(cols):
        expected=ks_2samp(train.loc[train.label==0,col],train.loc[train.label==1,col]).statistic
        check(np.isclose(expected,sep.loc[col,'KS_Stat']),f'Train-only KS {col}')
        w=wasserstein_distance(scaled[train.label==0,i],scaled[train.label==1,i])
        check(np.isclose(w,sep.loc[col,'Wasserstein_Std']),f'Train-only W {col}')
    runs=pd.read_csv('Model/tables/all_seed_runs.csv')
    impact=pd.read_csv('Model/tables/practical_impact.csv')
    for seed in [42,123,456,789,2026]:
        pred=pd.read_csv(f'Model/outputs/test_predictions_seed_{seed}.csv')
        check(pred.instance_id.tolist()==test.instance_id.tolist(),f'Shared evaluation IDs {seed}')
        for tier in [1,2,3]:
            row=runs[(runs.seed==seed)&runs.method.str.startswith(f'Tier {tier}')].iloc[0]
            score=pred[f'T{tier}_score'];yhat=pred[f'T{tier}_pred']
            values={'PR_AUC':average_precision_score(yt,score),'ROC_AUC':roc_auc_score(yt,score),
                'F1':f1_score(yt,yhat),'F2':fbeta_score(yt,yhat,beta=2),
                'Precision':precision_score(yt,yhat),'Recall':recall_score(yt,yhat)}
            for metric,expected in values.items():check(np.isclose(expected,row[metric]),f'Replayed {seed} T{tier} {metric}')
            check(np.array_equal(yhat,(score>=row.Threshold_tau).astype(int)),f'Threshold decision {seed} T{tier}')
            cm=confusion_matrix(yt,yhat,labels=[0,1]).ravel()
            ir=impact[(impact.seed==seed)&(impact.Tier==tier)].iloc[0]
            check(np.array_equal(cm,[ir.TN,ir.FP,ir.FN,ir.TP]),f'Impact confusion counts {seed} T{tier}')
            check(np.isclose(ir.Monthly_Reviews,ir.Monthly_Caught+ir.Monthly_False_Alerts),f'All alert reviews {seed} T{tier}')
            check(np.isclose(ir.Monthly_Cost,ir.Monthly_Escaped*1200+ir.Monthly_Reviews*30),f'Cost arithmetic {seed} T{tier}')
        v=pd.read_csv(f'Model/outputs/val_predictions_seed_{seed}.csv')
        candidates=np.percentile(v.T2_score,np.linspace(5,95,91))
        tau=max(candidates,key=lambda z:fbeta_score(v.label,v.T2_score>=z,beta=2,zero_division=0))
        check(np.isclose(tau,runs[(runs.seed==seed)&runs.method.str.startswith('Tier 2')].Threshold_tau.iloc[0]),f'Validation-only cutoff {seed}')
    # Fresh fit reproduces archived seed 42 scores and normal-only scaler statistics.
    clean=train.loc[train.label==0,cols].to_numpy()
    model=AutoencoderDetector(24,seed=42).fit(clean)
    check(np.allclose(model.scaler.mean_,clean.mean(axis=0)),'Normal-only fitted scaler')
    expected=pd.read_csv('Model/outputs/test_predictions_seed_42.csv')
    check(np.allclose(model.score_samples(test[cols].to_numpy()),expected.T2_score,rtol=1e-5),'Fresh seed42 model matches archived predictions')
    fits=json.loads(Path('Model/tables/ablation_fitting_ids.json').read_text())
    labels=full.set_index('instance_id').label
    for key,ids in fits.items():
        seed,k=map(int,key.split(':'))
        check(len(ids)==32 and len(set(ids))==32,'Fixed-size unique contamination rows')
        check(set(ids)<=sets['train'],'Contamination uses training only')
        check(labels.loc[ids].sum()==k,'Exact contamination count')
    ab=pd.read_csv('Model/tables/ablation_runs.csv')
    for seed in [42,123,456,789,2026]:
        baseline=ab[(ab.seed==seed)&(ab.experiment=='features')&(ab.setting=='Full')].iloc[0]
        rr=runs[(runs.seed==seed)&runs.method.str.startswith('Tier 2')].iloc[0]
        check(np.isclose(baseline.PR_AUC,rr.PR_AUC),'Full-mask control matches main AP')
        check(np.isclose(baseline.F2,rr.F2),'Full-mask control matches main F2')
    attr=pd.read_csv('Model/tables/feature_attribution_flagged_sample.csv')
    check(np.isclose(attr.Contribution_Percent.sum(),100),'Attribution sums to 100%')
    check(np.isclose(attr.Reconstruction_MSE.mean(),attr.Score.iloc[0]),'Attribution MSE recovers anomaly score')
    cases=pd.read_csv('Model/tables/failure_cases.csv')
    check(cases.instance_id.nunique()>=3,'Three distinct real error cases')
    check((cases.Type=='FP').any() and cases.Type.str.startswith('FN').any(),'FP and FN covered')
    for row in cases.itertuples():
        real=expected[expected.instance_id==row.instance_id].iloc[0]
        check(np.isclose(real.T2_score,row.score),'Case score matches archived prediction')
        check(np.isclose(row.score-row.threshold,row.distance_to_tau),'Case boundary distance')
    prefix='HW2_'+team['team_id']
    for suffix in ['report.pdf','report_anonymized.pdf']:
        f=Path(f'{prefix}_{suffix}');r=PdfReader(f)
        check(f.read_bytes()==Path('Paper',f.name).read_bytes(),'Root/Paper identical '+suffix)
        text='\n'.join(p.extract_text() or '' for p in r.pages)
        check('Contributions of Each Member' in '\n'.join(p.extract_text() or '' for p in r.pages[:6]),'Contributions inside page six')
        check('AI Usage Statement' in '\n'.join(p.extract_text() or '' for p in r.pages[:6]),'AI disclosure inside page six')
        check('References' in text,'Real reference section')
        check(not re.search(r'\?\?|\[\?\]',text),'Resolved references')
        check(all(len(p.extract_text() or '')>50 for p in r.pages),'Selectable PDF text')
        if 'anonymized' in suffix:
            check(all(m['student_id'] not in text and m['name'] not in text for m in team['members']),'Anonymous PDF hides member identifiers')
            check('Team G04' in text,'Anonymous PDF has team ID')
        else:
            first=r.pages[0].extract_text()
            check(team['repository_url'] in first.replace('\n',''),'First-page repository URL')
            check(all(m['student_id'] in first for m in team['members']),'All five student IDs on first page')
        # References are the only permitted content after page six.
        if len(r.pages)>6:
            check((r.pages[6].extract_text() or '').lstrip().startswith('References'),'Only references start page seven')
    pptx=Path(prefix+'_presentation.pptx')
    check(pptx.read_bytes()==Path('Presentation',pptx.name).read_bytes(),'Root/Presentation identical deck')
    with ZipFile(pptx) as z:
        slides=[n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',n)]
        check(len(slides)==11,'Eleven presentation slides')
        media=[n for n in z.namelist() if n.startswith('ppt/media/')]
        check(len(media)>=4,'Chart images embedded in PPTX')
        check(len([n for n in z.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml',n)])==11,'Every slide has speaker notes')
    receipt=json.loads(Path('Presentation/validation.json').read_text())
    check(hashlib.sha256(pptx.read_bytes()).hexdigest()==receipt['finalSha256'],'Deck matches validation receipt')
    check(receipt['presentationLayout']['finding_count']==0,'Slide layout validator clean')
    check(receipt['packageIntegrity']['finding_count']==0,'Slide package validator clean')
    content=json.loads(Path('Presentation/content.json').read_text())
    check(sum(s['seconds'] for s in content)==480,'Speaker budget totals 480 seconds')
    with ZipFile(prefix+'.zip') as z:
        names=[prefix+'_report.pdf',prefix+'_presentation.pptx']
        check(sorted(z.namelist())==sorted(names),'Exact E3 names without directories')
        check(z.testzip() is None,'ZIP CRC integrity')
        for name in names:check(z.read(name)==Path(name).read_bytes(),'ZIP contains latest '+name)
    pending=[m['name'] for m in team['members'] if 'pending' in m['contribution'].lower()]
    result={'checks_passed':checks,'technical_verification':'pass',
        'remaining_human_items':['Confirm member contributions','Team review and 8-minute live rehearsal','Verify public repository access','Upload to E3 and retain receipt'],
        'contributions_pending':pending,'external_submission_performed':False,'grade_claimed':False}
    Path('docs/verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
