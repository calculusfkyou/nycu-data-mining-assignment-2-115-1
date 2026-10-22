"""Legible single-column report figures, sized at their final printed width."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve,average_precision_score
import paper_style as style

def main():
    style.apply()
    train=pd.read_csv('Dataset/train.csv')
    fig,axes=plt.subplots(1,2,figsize=(3.4,2.05),dpi=300)
    for ax,col,title in zip(axes,['tokens_received','patch_char_length'],['Received tokens','Patch length']):
        groups=[np.log1p(train.loc[train.label==y,col].to_numpy()) for y in [0,1]]
        ax.boxplot(groups,tick_labels=['Normal','Failure'],showfliers=False,widths=.5)
        ax.set_title(title,fontsize=8);ax.set_ylabel('log(1 + count)',fontsize=7)
        ax.tick_params(labelsize=7)
    fig.tight_layout(pad=.5,w_pad=.8);style.save(fig,Path('Paper/figures/fig_training_compact'))
    pred=pd.read_csv('Model/outputs/test_predictions.csv')
    fig,ax=plt.subplots(figsize=(3.4,2.0),dpi=300)
    for tier,color in [(1,style.REF_GRAY),(2,style.BLUE_MAIN),(3,style.RED)]:
        pr,re,_=precision_recall_curve(pred.label,pred[f'T{tier}_score'])
        ap=average_precision_score(pred.label,pred[f'T{tier}_score'])
        ax.plot(re,pr,label=f'T{tier}: AP {ap:.3f}',color=color,lw=1.1)
    ax.axhline(pred.label.mean(),color=style.REF_GRAY,linestyle=':',lw=.8,label=f'Prior {pred.label.mean():.3f}')
    ax.set_xlabel('Failure recall');ax.set_ylabel('Precision');ax.legend(fontsize=7,loc='lower left')
    fig.tight_layout(pad=.5);style.save(fig,Path('Paper/figures/fig_pr_compact'))
    attr=pd.read_csv('Model/tables/feature_attribution_flagged_sample.csv').head(4)
    fig,ax=plt.subplots(figsize=(3.4,1.8),dpi=300)
    names={'stalled_turn_count':'Stalled turns','reproduce_script_created':'Reproduction flag','patch_num_files_changed':'Changed files','patch_hunk_count':'Patch hunks'}
    bars=ax.barh([names[x] for x in attr.Feature],attr.Contribution_Percent,color=style.BLUE_MAIN,height=.55)
    for bar,pct in zip(bars,attr.Contribution_Percent):ax.text(pct+1,bar.get_y()+bar.get_height()/2,f'{pct:.1f}%',va='center',fontsize=7)
    ax.invert_yaxis();ax.set_xlim(0,103);ax.set_xlabel('Reconstruction contribution (%)');ax.tick_params(axis='y',labelsize=7)
    fig.tight_layout(pad=.5);style.save(fig,Path('Paper/figures/fig_attr_compact'))

if __name__=='__main__': main()
