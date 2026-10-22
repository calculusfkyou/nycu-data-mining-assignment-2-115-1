"""Build v2 publication tables, final-size figures and slide content from saved evidence."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

T=Path('Model/tables/v2'); G=Path('Paper/generated'); F=Path('Paper/figures/v2')
F.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
LABELS={'always_alert':'Always alert','never_alert':'Never alert','heuristic_T1':'T1: heuristic',
 'knn_R3_T2':'T2: dual encoder kNN','supervised_R3_logistic_T3':'T3: encoder logistic',
 'supervised_R1_logistic':'TF-IDF logistic','inductive_ecod_R0_clean':'Inductive ECOD variant',
 'isolation_forest_R0_clean':'Isolation Forest','conditional_residual_R0_clean':'Conditional residual',
 'shallow_dae_R0_clean':'Denoising AE','legacy_ae_R0':'Numeric AE (v2 control)', 'legacy_xgboost_R0':'XGBoost (v2 control)'}
def save(fig,name):
    fig.savefig(F/(name+'.pdf'),bbox_inches='tight');fig.savefig(F/(name+'.png'),dpi=190,bbox_inches='tight');plt.close(fig)
def main():
    df=pd.read_csv(T/'main_results.csv');oof=pd.read_csv('Model/outputs/v2/oof_predictions.csv')
    folds=json.loads(Path('Dataset/v2/splits.json').read_text())['outer_folds']; indexed=oof.set_index('instance_id')
    rows=[]; foldrows=[]
    for r in df.itertuples():
        prefix=f'{r.model}_seed_{r.seed}'; raw=indexed[prefix+'_score']
        for fold in folds:
            z=indexed.loc[fold['test_instance_ids']]; y=z.true_label; s=z[prefix+'_score']; pred=z[prefix+'_pred']
            tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
            foldrows.append(dict(model=r.model,seed=r.seed,fold=fold['fold'],tn=tn,fp=fp,fn=fn,tp=tp,
                balanced_accuracy=.5*(tn/(tn+fp)+tp/(tp+fn)),roc_auc=roc_auc_score(y,s),average_precision=average_precision_score(y,s)))
    fr=pd.DataFrame(foldrows);fr.to_csv(T/'publication_fold_metrics.csv',index=False)
    for model in LABELS:
        r=df[(df.model==model)&(df.seed==42)].iloc[0]
        f=fr[(fr.model==model)&(fr.seed==42)]
        rows.append(f"{LABELS[model]} & {r.balanced_accuracy:.3f} & {r.fpr:.3f} & {r.failure_recall:.3f} & {int(r.tn)}/{int(r.fp)}/{int(r.fn)}/{int(r.tp)} & {f.average_precision.mean():.3f} "+r'\\')
    (G/'v2_main.tex').write_text('\n'.join(rows)+'\n')
    rows=[]
    for model in ['knn_R3_T2','isolation_forest_R0_clean','legacy_ae_R0','shallow_dae_R0_clean']:
        z=df[df.model==model];ff=fr[(fr.model==model)&(fr.seed==42)]
        seedstd='--' if len(z)==1 else f'{z.balanced_accuracy.std(ddof=1):.3f}'
        rows.append(f'{LABELS[model]} & {ff.balanced_accuracy.mean():.3f} $\\pm$ {ff.balanced_accuracy.std(ddof=1):.3f} & {seedstd} '+r'\\')
    (G/'v2_variation.tex').write_text('\n'.join(rows)+'\n')
    rep=pd.read_csv(T/'ablation_representations.csv');ch=pd.read_csv(T/'ablation_channels.csv')
    names=['R0-clean numeric','R1 TF-IDF','R2 MiniLM logs','R3 dual encoders']
    rows=[f'{name} & {r.balanced_accuracy:.3f} & {r.fpr:.3f} & {r.failure_recall:.3f} '+r'\\' for name,r in zip(names,rep.itertuples())]
    patch=ch[ch.channel=='patch_only_codebert'].iloc[0]
    rows.append(f'CodeBERT patch only & {patch.balanced_accuracy:.3f} & {patch.fpr:.3f} & {patch.failure_recall:.3f} '+r'\\')
    (G/'v2_ablation.tex').write_text('\n'.join(rows)+'\n')
    # EDA is restricted to one designated outer-training partition, never pooled held-out rows.
    raw=pd.read_csv('Dataset/sweagent_gpt4_features.csv').set_index('instance_id');train=raw.loc[folds[0]['train_instance_ids']]
    stats=[]
    for c in ['num_steps','num_error_observations','tokens_received','patch_char_length']:
        a=train.loc[train.label==0,c]; b=train.loc[train.label==1,c]
        label={'num_steps':'Steps','num_error_observations':'Error observations','tokens_received':'Received tokens','patch_char_length':'Patch characters'}[c]
        stats.append(f'{label} & {a.median():.0f} & {b.median():.0f} '+r'\\')
    (G/'v2_stats.tex').write_text('\n'.join(stats)+'\n')
    fig,ax=plt.subplots(1,2,figsize=(3.35,1.75))
    for a,c,label in zip(ax,['tokens_received','patch_char_length'],['Received tokens','Patch characters']):
        a.boxplot([np.log1p(train.loc[train.label==y,c]) for y in [0,1]],tick_labels=['Success','Failure'],showfliers=False,widths=.55)
        a.set_title(label,fontsize=9);a.tick_params(labelsize=8);a.set_ylabel('log(1 + count)',fontsize=8)
    fig.tight_layout(pad=.5);save(fig,'eda')
    fig,axs=plt.subplots(1,3,figsize=(6.8,2.0))
    for ax,model in zip(axs,['knn_R3_T2','inductive_ecod_R0_clean','conditional_residual_R0_clean']):
        r=df[(df.model==model)&(df.seed==42)].iloc[0]; cm=np.array([[r.tn,r.fp],[r.fn,r.tp]],int)
        ax.imshow(cm/cm.sum(axis=1,keepdims=True),vmin=0,vmax=1,cmap='Blues')
        for i in range(2):
            for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=13,color='white' if cm[i,j]/cm[i].sum()>.5 else '#15283B')
        ax.set_xticks([0,1],['No alert','Alert']);ax.set_yticks([0,1],['Success','Failure']);ax.tick_params(labelsize=8)
        ax.set_title(LABELS[model].replace('T2: dual encoder kNN','T2: dual encoder'),fontsize=10);ax.set_xlabel('Predicted');ax.set_ylabel('Actual')
    fig.tight_layout(pad=.5);save(fig,'confusion')
    fig,ax=plt.subplots(figsize=(3.35,2.0)); vals=list(rep.balanced_accuracy)+[patch.balanced_accuracy]
    ax.barh(['Numeric','TF-IDF','Log encoder','Dual encoders','Patch encoder'],vals,color=['#596D83','#596D83','#84A7CA','#0F4D92','#84A7CA'])
    ax.axvline(.5,ls='--',c='black',lw=.8);ax.set_xlim(0,.8);ax.set_xlabel('Balanced accuracy');ax.invert_yaxis();fig.tight_layout(pad=.5);save(fig,'representations')
    con=pd.read_csv(T/'ablation_contamination.csv'); fig,ax=plt.subplots(figsize=(3.35,1.9))
    for mode,label in [('clean_tau','Fixed clean threshold'),('recalibrated_tau','Recalibrated threshold')]:
        z=con[con.threshold_mode==mode].groupby('p_contamination').balanced_accuracy.agg(['mean','std'])
        ax.errorbar(z.index*100,z['mean'],yerr=z['std'],marker='o',capsize=3,label=label)
    ax.set_xlabel('Requested replacement (%)');ax.set_ylabel('Balanced accuracy');ax.legend(fontsize=7);fig.tight_layout(pad=.5);save(fig,'contamination')
    cg=con.groupby(['p_contamination','threshold_mode']).balanced_accuracy.agg(['mean','std']).reset_index()
    (G/'v2_contamination.tex').write_text('\n'.join(f'{r.p_contamination*100:.0f} & {r.threshold_mode.replace("_", " ")} & {r.mean:.3f} $\\pm$ {r.std:.3f} '+r'\\' for r in cg.itertuples())+'\n')
    # Reusable content for the established editable-slide builder.
    slides=[]
    def add(title,lines,sec,notes,image=None,table=None):slides.append(dict(title=title,lines=lines,seconds=sec,notes=notes,image=image,table=table))
    add('Can local text embeddings detect agent failures?', ['Team G04 · NYCU Data Mining HW2','284 real SWE-agent runs from SWE-bench Lite','MiniLM logs + CodeBERT patches, computed locally','Frozen encoders lose to simple numeric controls'],30,'Task-end failure screening. No generated trajectories or new LLM API calls. Official resolution labels are the target. Source: Dataset/evaluation_results.json. Full author list and student IDs appear in the report.')
    add('The decision is whether a completed run needs review',['Normal reference: 54 officially resolved tasks','Failure label: 230 unresolved tasks (81.0%)','Errors, large patches and high cost do not prove failure','Failure is common here, which limits the anomaly framing'],40,'Category 4: Text / Logs. Labels do not establish deception, reward hacking or false completion. No prefix prediction or autonomous release has been tested. SWE-bench and SWE-agent sources are in Paper/custom.bib.')
    add('Training-only analysis shows overlapping behavior',['EDA uses outer fold 0 training rows only',f'{sum(train.label==0)} successful and {sum(train.label==1)} failed training runs','Long tails motivate log1p and robust scaling','Remove duplicated command count from numeric features'],35,'No held-out row enters these boxplots. Numeric control has 23 columns, including 4 binary flags. The test-string flag is patch-wide and the command heuristic is noisy. Source: Paper/figures/v2/eda and v2_stats.tex.',image='Paper/figures/v2/eda.png')
    add('Local encoders preserve more of the recorded text',['MiniLM: chunk logs, average within and across steps','CodeBERT: chunk diff hunks, pool token representations','Normalize 384-D log and 768-D patch vectors','T2: mean distance to k nearby successful training runs'],45,'Encoders remain frozen. Log chunks:224 tokens,32 overlap. Patch chunks:480 tokens,64 overlap. Log and patch cosine distances have equal weight. k in {1,3,5}; only successful training rows form the reference. Pooling loses ordering and does not verify correctness.')
    add('Every method shares five outer folds',['Approx. 60% training, 20% validation, 20% outer test','Validation selects model and threshold; never fits weights','Choose recall under validation FPR <=20%','284 out-of-fold predictions per model and seed','Prior data inspection makes this exploratory evaluation'],45,'Only around 32 successful training and 10 successful validation runs per fold. Empirical validation FPR is not a test guarantee. Shared repositories remain; additional held-out-repository sensitivity is limited. Deterministic methods run once, random methods have seeds42,123,456.')
    values=[['Method','Bal. acc.','FPR','Recall','Unflagged OK']]
    for model in ['knn_R3_T2','supervised_R3_logistic_T3','supervised_R1_logistic','inductive_ecod_R0_clean','conditional_residual_R0_clean']:
        r=df[(df.model==model)&(df.seed==42)].iloc[0];values.append([{'knn_R3_T2':'T2: embeddings','supervised_R3_logistic_T3':'T3: supervised','supervised_R1_logistic':'TF-IDF supervised','inductive_ecod_R0_clean':'ECOD variant','conditional_residual_R0_clean':'Conditional'}[model],f'{r.balanced_accuracy:.3f}',f'{r.fpr:.3f}',f'{r.failure_recall:.3f}',f'{r.unflagged_success_fraction:.1%}'])
    add('Numeric controls outperform the embedding detector',['Seed 42 pooled counts; no independent deployment test','ECOD minus T2 balanced accuracy: +0.176','Paired conditional 95% interval: [0.114, 0.240]'],55,'All numbers derive from main_results.csv and bootstrap_ci.csv. Bootstrap resamples paired OOF instances within class2000times; shared training folds limit interpretation. Conditional residual is a secondary comparison. No method meets all operational gates.',table=values)
    add('The confusion matrices expose the missed failures',['T2: 179 of 230 failures receive no alert','Only 40 of 219 unflagged runs actually succeed','Better numeric scores still leave many failures unflagged'],50,'Rows are actual success/failure. Columns are no alert/alert. Cell text is raw count; color is row fraction. T2 TN40FP14FN179TP51;ECOD TN47FP7FN128TP102;conditional TN44FP10FN107TP123. Prediction absence of alert is not a correctness guarantee.',image='Paper/figures/v2/confusion.png')
    add('Patch vectors help, but dual-channel averaging hurts',['Numeric kNN: balanced accuracy 0.641','Patch-only encoder: 0.594','Dual encoder: 0.481; log-only: 0.456','H1 unsupported; patch addition helps logs, not patch-only'],50,'All use kNN and validation model selection. Distances differ across representations, so this comparison includes geometry. Patch embeddings may encode surface similarity rather than functional correctness. Equal averaging is one tested fusion rule, not proof all text encoders fail.',image='Paper/figures/v2/representations.png')
    means=cg[cg.threshold_mode=='clean_tau']
    add('Contamination does not show a consistent penalty',['Fixed reference size; training failures replace successes','Freeze each clean fold’s chosen k and threshold','Also report validation-only recalibration','Zero-contamination reproduces the primary detector'],40,'Corrected after detecting a mismatched k=3 control in the earlier v2 draft. Fitting IDs and exact replacement fractions are archived. Three replacement seeds. H3 is not established if effects are small or nonmonotonic. Source:contamination_fitting_ids.json and ablation_contamination.csv.',image='Paper/figures/v2/contamination.png')
    add('Near-normal vectors can still describe failed patches',['FP django-13230: score .11276, threshold .10865','FN sympy-12454: score .02483, threshold .07315','Boundary FP sympy-24213: margin +.00018','Distances identify similarity, not the semantic cause'],50,'Full IDs and raw SHA256s are in case_studies.json. Nearest reference log/patch distances respectively .20290/.00655, .04656/.00311, .14414/.00251. The saved leave-one-out values use a fixed nearest reference, not full kNN rescoring; do not call them complete score attribution.')
    add('These detectors cannot justify automatic release',['No model meets the predeclared practical gates','Next: fresh agent runs and task-conditioned evidence','Encoding 284 runs took about 895 seconds on CPU','All five member contributions remain pending confirmation'],40,'Gates:recall>=90%,FPR<=20%,unflagged success>=80%,coverage>=5%. Encoder memory reported ~1.19GiB by extractor profiling; not a whole-process-tree memory guarantee. Actual live rehearsal and E3 upload remain outstanding. AI assistance and all five author IDs appear in the report.')
    assert len(slides)==11 and sum(x['seconds'] for x in slides)==480
    Path('Presentation/content.json').write_text(json.dumps(slides,indent=2)+'\n')
    Path('Presentation/speaker_notes.md').write_text('# V2 eight-minute rehearsal guide\n\n'+'\n\n'.join(f"## {i+1}. {s['title']} ({s['seconds']} seconds)\n\n{s['notes']}" for i,s in enumerate(slides))+'\n')
    print('Generated v2 tables, five figures, and 11-slide/480-second content.')

if __name__=='__main__':main()
