"""Independently replay v2 evidence and check the exact deliverable package."""
import hashlib,json,re,sys
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from pypdf import PdfReader
from sklearn.metrics import confusion_matrix,roc_auc_score,average_precision_score
from Model.models_v2 import KnnNormalDistanceDetector,select_validation_threshold
from Dataset.download_and_extract import extract_features_from_trajectory

N=0
def check(ok,message):
    global N
    if not bool(ok):raise AssertionError(message)
    N+=1
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    team=json.loads(Path('Paper/team.json').read_text(encoding='utf-8-sig'))
    raw=pd.read_csv('Dataset/sweagent_gpt4_features.csv').set_index('instance_id')
    truth=json.loads(Path('Dataset/evaluation_results.json').read_text());resolved=set(truth['resolved'])
    provenance=json.loads(Path('Dataset/provenance.json').read_text())['trajectories']
    check(set(raw.index)==set(truth['generated']),'Official generated coverage')
    for item in provenance:
        iid=item['instance_id'];p=Path('Dataset/raw/sweagent_gpt4')/(iid+'.traj')
        check(sha(p)==item['sha256'],'Raw SHA '+iid)
        check(raw.loc[iid,'label']==int(iid not in resolved),'Official label '+iid)
        rebuilt=extract_features_from_trajectory(json.loads(p.read_bytes()),iid,iid in resolved)
        columns=raw.columns[4:]
        check(np.allclose(raw.loc[iid,columns].to_numpy(float),[rebuilt[x] for x in columns]),'Raw features '+iid)
    folds=json.loads(Path('Dataset/v2/splits.json').read_text())['outer_folds'];alltest=[]
    for f in folds:
        a,b,c=[set(f[x+'_instance_ids']) for x in ['train','val','test']]
        check(not(a&b or a&c or b&c),'Fold disjoint')
        check(a|b|c==set(raw.index),'Fold coverage')
        check(set(f['train_normal_instance_ids'])==a&resolved,'Exact normal fit IDs')
        alltest+=f['test_instance_ids']
    check(len(alltest)==284 and len(set(alltest))==284,'Each run tested once')
    oof=pd.read_csv('Model/outputs/v2/oof_predictions.csv').set_index('instance_id')
    check(set(oof.index)==set(raw.index) and oof.index.is_unique,'OOF exact IDs')
    check(np.array_equal(oof.true_label,raw.loc[oof.index,'label']),'OOF labels')
    results=pd.read_csv('Model/tables/v2/main_results.csv')
    for r in results.itertuples():
        stem=f'{r.model}_seed_{r.seed}';s=oof[stem+'_score'];p=oof[stem+'_pred'];y=oof.true_label
        check(np.isfinite(s).all() and set(p.unique())<={0,1},'Finite model '+stem)
        tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
        check([tn,fp,fn,tp]==[r.tn,r.fp,r.fn,r.tp],'Replay counts '+stem)
        for actual,saved,name in [(.5*(tn/(tn+fp)+tp/(tp+fn)),r.balanced_accuracy,'BA'),(fp/(tn+fp),r.fpr,'FPR'),(tp/(tp+fn),r.failure_recall,'recall'),(roc_auc_score(y,s),r.roc_auc,'AUC'),(average_precision_score(y,s),r.average_precision,'AP')]:check(np.isclose(actual,saved),'Replay '+stem+' '+name)
        gates=(r.failure_recall>=.9 and r.fpr<=.2 and r.unflagged_success_fraction>=.8 and r.unflagged_coverage>=.05)
        check(not gates,'No unsupported operational pass '+stem)
    # Re-score primary on frozen embeddings; reselect k/tau using validation only.
    log=np.load('Dataset/v2/embeddings/r2_minilm.npz');patch=np.load('Dataset/v2/embeddings/r3_codebert.npz')
    ids=oof.index.tolist();ix={x:i for i,x in enumerate(ids)}
    lm={str(x):i for i,x in enumerate(log['instance_ids'])};pm={str(x):i for i,x in enumerate(patch['instance_ids'])}
    a=np.array([log['embeddings'][lm[x]] for x in ids]);b=np.array([patch['embeddings'][pm[x]] for x in ids]);e=np.array([patch['is_empty'][pm[x]] for x in ids])
    check(a.shape==(284,384) and b.shape==(284,768),'Encoder dimensions')
    check(np.isfinite(a).all() and np.isfinite(b).all(),'Finite embeddings')
    params=pd.read_csv('Model/tables/v2/fold_parameters.csv')
    for f in folds:
        ri=[ix[x] for x in f['train_normal_instance_ids']];vi=[ix[x] for x in f['val_instance_ids']];ti=[ix[x] for x in f['test_instance_ids']]
        par=params[(params.fold==f['fold'])&(params.model=='knn_R3_T2')].iloc[0];candidates=[]
        for k in [1,3,5]:
            model=KnnNormalDistanceDetector(k=k).fit_embeddings(a[ri],b[ri],e[ri]);v=model.score_dual(a[vi],b[vi],e[vi])
            tau,m=select_validation_threshold(oof.true_label.iloc[vi].to_numpy(),v);candidates.append((m['balanced_accuracy'],-k,tau))
        _,negativek,tau=max(candidates);k=-negativek
        check(k==int(par.chosen_param.split('=')[1]) and np.isclose(tau,par.tau),'Validation-selected primary parameters')
        model=KnnNormalDistanceDetector(k=k).fit_embeddings(a[ri],b[ri],e[ri]);s=model.score_dual(a[ti],b[ti],e[ti])
        check(np.allclose(s,oof.knn_R3_T2_seed_42_score.iloc[ti],atol=1e-7),'Replayed primary scores')
        check(np.array_equal(s>=tau,oof.knn_R3_T2_seed_42_pred.iloc[ti]),'Replayed threshold')
    records=json.loads(Path('Model/tables/v2/contamination_fitting_ids.json').read_text())
    for r in records:
        f=folds[r['fold']];fits=r['fitting_ids'];train=set(f['train_instance_ids'])
        check(len(fits)==len(set(fits))==r['n_reference'],'Fixed reference size')
        check(set(fits)<=train,'No calibration contamination')
        check(raw.loc[fits,'label'].sum()==r['n_replaced'],'Exact injected labels')
        check(r['n_replaced']==int(np.floor(r['n_reference']*r['p_contamination'])),'Exact actual fraction')
    cp=pd.read_csv('Model/outputs/v2/contamination_predictions.csv')
    cr=pd.read_csv('Model/tables/v2/ablation_contamination.csv')
    for r in cr.itertuples():
        z=cp[(cp.seed==r.seed)&(cp.p_contamination==r.p_contamination)]
        mode='fixed' if r.threshold_mode=='clean_tau' else 'recalibrated';p=z[mode+'_pred']
        check(z.instance_id.nunique()==284,'Contamination OOF coverage')
        check(np.array_equal(p,(z.score>=z['clean_tau' if mode=='fixed' else 'recalibrated_tau']).astype(int)),'Contamination decision')
        check(np.array_equal(confusion_matrix(z.label,p,labels=[0,1]).ravel(),[r.tn,r.fp,r.fn,r.tp]),'Contamination counts')
        if r.p_contamination==0:
            check(np.array_equal(p.to_numpy(),oof.loc[z.instance_id,'knn_R3_T2_seed_42_pred'].to_numpy()),'Zero contamination control')
    cases=json.loads(Path('Model/tables/v2/case_studies.json').read_text())
    for c in cases.values():
        if not c.get('found'):continue
        real=oof.loc[c['instance_id']]
        check(c['official_label']==real.true_label and c['prediction']==real.knn_R3_T2_seed_42_pred,'Case labels/prediction')
        check(np.isclose(c['score'],real.knn_R3_T2_seed_42_score),'Case score')
        check(sha(Path('Dataset/raw/sweagent_gpt4')/(c['instance_id']+'.traj'))==c['sha256'],'Case SHA')
    for name in ['HW2_G04_report.pdf','HW2_G04_report_anonymized.pdf']:
        check(Path(name).read_bytes()==Path('Paper',name).read_bytes(),'Root/Paper copy')
        pdf=PdfReader(name);texts=[p.extract_text() or '' for p in pdf.pages];ref=next(i for i,t in enumerate(texts) if t.lstrip().startswith('References'))
        check(ref<=6 and all('Contributions of Each Member' not in t for t in texts[ref:]),'Body page limit')
        body='\n'.join(texts[:ref]);check('Contributions of Each Member' in body and 'AI Usage Statement' in body,'Required end matter')
        check('179' in body and '0.481' in body and 'CodeBERT' in body,'Updated v2 findings')
        check(not re.search(r'\?\?|\[\?\]',body),'Resolved citations')
        if 'anonymized' in name:check(all(m['name'] not in body and m['student_id'] not in body for m in team['members']),'Anonymity')
        else:
            check(all(m['student_id'] in texts[0] for m in team['members']),'First page IDs')
            check(team['repository_url'] in texts[0].replace('\n',''),'First page GitHub URL')
    deck=Path('HW2_G04_presentation.pptx');check(deck.read_bytes()==Path('Presentation',deck.name).read_bytes(),'Deck copies')
    with ZipFile(deck) as z:
        slides=sorted(n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',n));check(len(slides)==11,'11 slides')
        check(len([n for n in z.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml',n)])==11,'11 notes')
        text=' '.join(' '.join(ET.fromstring(z.read(n)).itertext()) for n in slides)
        check('179' in text and 'CodeBERT' in text and '0.481' in text,'V2 deck evidence')
        check(sum(n.startswith('ppt/media/') for n in z.namelist())>=4,'Embedded figures')
    receipt=json.loads(Path('Presentation/validation.json').read_text())
    check(sha(deck)==receipt['finalSha256'],'Validated deck hash')
    check(receipt['packageIntegrity']['finding_count']==receipt['presentationLayout']['finding_count']==0,'Deck structural/layout checks')
    check(sum(s['seconds'] for s in json.loads(Path('Presentation/content.json').read_text()))==480,'8-minute budget')
    with ZipFile('HW2_G04.zip') as z:
        names=['HW2_G04_report.pdf','HW2_G04_presentation.pptx'];check(sorted(z.namelist())==sorted(names),'Exact ZIP paths')
        check(z.testzip() is None,'ZIP CRC')
        for name in names:check(z.read(name)==Path(name).read_bytes(),'Latest ZIP bytes')
    result=dict(experiment_version='v2',checks_passed=N,technical_verification='pass',body_pages=ref,
        files={name:sha(name) for name in ['HW2_G04_report.pdf','HW2_G04_report_anonymized.pdf','HW2_G04_presentation.pptx','HW2_G04.zip']},
        remaining_human_items=['Confirm member contributions','Team review and actual 8-minute rehearsal','Verify public repository access','Upload to E3 and retain receipt'],external_submission_performed=False,grade_claimed=False)
    Path('docs/verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
