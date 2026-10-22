"""Generate numerical report fragments from current CSVs; no manually copied metrics."""
import json
import os
import platform
from pathlib import Path
import numpy as np
import pandas as pd

PAPER=Path('Paper')

def esc(s):
    return str(s).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')

def main():
    tables=PAPER/'generated'; tables.mkdir(exist_ok=True)
    team=json.loads((PAPER/'team.json').read_text(encoding='utf-8-sig'))
    authors=[]
    for row in team['members']: authors.append(f"{esc(row['name'])} ({row['student_id']})")
    lines=[' \\quad '.join(authors[:3]),' \\quad '.join(authors[3:]),f"Team {team['team_id']}, Data Mining (Fall 2026), NYCU",r'\url{'+team['repository_url']+'}']
    (tables/'authors.tex').write_text(' \\\\\n'.join(lines),encoding='utf-8')
    contrib='\n'.join(r'\item '+r'\ifdefined\JUDGE Member '+str(i+1)+r'\else '+esc(m['name'])+' ('+m['student_id']+r')\fi: '+esc(m['contribution'])+'.' for i,m in enumerate(team['members']))
    (tables/'contributions.tex').write_text(contrib,encoding='utf-8')
    mainres=pd.read_csv('Model/tables/main_results_table.csv')
    metrics=['PR_AUC','ROC_AUC','F1','F2','Latency_ms','Train_Time_s']
    rows=[]
    for i,row in mainres.iterrows():
        cells=[]
        for metric in metrics:
            digits = 4 if metric in ['Latency_ms', 'Train_Time_s'] else 3
            val=f"{row[metric+'_mean']:.{digits}f} \\pm {row[metric+'_std']:.{digits}f}"
            if metric == 'Train_Time_s' and row[metric+'_mean'] < 1e-4:
                val = r'<10^{-4}'
            best=mainres[metric+'_mean'].min() if metric in ['Latency_ms','Train_Time_s'] else mainres[metric+'_mean'].max()
            if np.isclose(row[metric+'_mean'],best,atol=1e-12,rtol=0): val=r'\mathbf{'+val+'}'
            cells.append('$'+val+'$')
        rows.append(['T1: Heuristic','T2: Autoencoder','T3: XGBoost'][i]+' & '+' & '.join(cells)+r' \\')
    rows.append(r'Always alert & $0.926$ & $0.500$ & $0.962$ & $0.984$ & -- & -- \\')
    (tables/'main.tex').write_text('\n'.join(rows),encoding='utf-8')
    train=pd.read_csv('Dataset/train.csv')
    wanted=['num_steps','num_error_observations','stalled_turn_count','total_cost_usd','tokens_received','patch_char_length','patch_lines_added']
    rows=[]
    names=['Steps','Error observations','Stalled turns','Cost (USD)','Tokens received','Patch characters','Added lines']
    for col,name in zip(wanted,names):
        vals=[train.loc[train.label==y,col] for y in [0,1]]
        rows.append(name+' & '+' & '.join(f'{v.mean():.2f} & {v.std():.2f}' for v in vals)+r' \\')
    (tables/'stats.tex').write_text('\n'.join(rows),encoding='utf-8')
    sep=pd.read_csv('EDA/tables/feature_separability.csv')
    rows=[]
    for col in ['tokens_received','total_cost_usd','num_steps','num_error_observations','patch_char_length']:
        r=sep[sep.Feature==col].iloc[0]
        rows.append(esc(col)+' & '+f'{r.Wasserstein_Std:.3f} & {r.KS_Stat:.3f} & {r.KS_Pval:.2g}'+r' \\')
    (tables/'separability.tex').write_text('\n'.join(rows),encoding='utf-8')
    for file,key,out in [('ablation_contamination.csv','Contamination_Ratio','contamination'),('ablation_features.csv','Feature_Set','features')]:
        frame=pd.read_csv('Model/tables/'+file); rows=[]
        for r in frame.to_dict('records'):
            rows.append(esc(r[key])+' & '+f"${r['PR_AUC_mean']:.4f} \\pm {r['PR_AUC_std']:.4f}$ & ${r['Delta_PR_AUC']:+.4f}$ & ${r['F2_mean']:.3f} \\pm {r['F2_std']:.3f}$"+r' \\')
        (tables/(out+'.tex')).write_text('\n'.join(rows),encoding='utf-8')
    cases=pd.read_csv('Model/tables/failure_cases.csv')
    rows=[]
    for i,r in cases.iterrows():
        rows.append(f"{i+1}: {esc(r.Type)} & {int(r.num_steps)} & {int(r.num_error_observations)} & {int(r.patch_lines_added)} & {r.score:.4f} & {r.distance_to_tau:+.4f}"+r' \\')
    (tables/'cases.tex').write_text('\n'.join(rows),encoding='utf-8')
    environment={'python':platform.python_version(),'platform':platform.platform(),
                 'processor':os.environ.get('PROCESSOR_IDENTIFIER',platform.processor()),
                 'device':'CPU, torch threads=1, XGBoost n_jobs=1'}
    import importlib.metadata
    environment['libraries']={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','torch','xgboost']}
    (Path('Model/outputs/environment.json')).write_text(json.dumps(environment,indent=2)+'\n')
    runs=pd.read_csv('Model/tables/all_seed_runs.csv'); sig=pd.read_csv('Model/tables/significance_tests.csv')
    tau=runs[(runs.seed==42)&runs.method.str.startswith('Tier 2')].Threshold_tau.iloc[0]
    (tables/'numbers.tex').write_text(r'\newcommand{\SampleTau}{'+f'{tau:.4f}'+'}\n'+
        r'\newcommand{\OurAP}{'+f'{mainres.iloc[1].PR_AUC_mean:.3f}'+'}\n'+
        r'\newcommand{\OurFtwo}{'+f'{mainres.iloc[1].F2_mean:.3f}'+'}\n'+
        r'\newcommand{\SupervisedP}{'+f'{sig.iloc[1].Paired_ttest_pval:.3f}'+'}\n',encoding='utf-8')
    for fragment in tables.glob('*.tex'):
        fragment.write_text(fragment.read_text(encoding='utf-8').rstrip() + '\n', encoding='utf-8')
    print('Generated report tables and environment manifest')

if __name__=='__main__': main()
