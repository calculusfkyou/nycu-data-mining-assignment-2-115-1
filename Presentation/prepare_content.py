"""Create the narrative and numbers used by the editable JavaScript slide builder."""
import json
from pathlib import Path
import pandas as pd

team=json.loads(Path('Paper/team.json').read_text(encoding='utf-8-sig'))
r=pd.read_csv('Model/tables/main_results_table.csv')
slides=[]
def add(title,lines,seconds,image=None,table=None,notes=''):
    slides.append(dict(title=title,lines=lines,seconds=seconds,image=image,table=table,notes=notes))
add('Screening coding-agent task failures',[
    'Normal-only reconstruction on real SWE-bench Lite trajectories',
    'Team G04 | Data Mining, NYCU | Fall 2026',
    'Tun-Chieh Yang · Jia-Cheng Wu · Min-Quan Wang',
    'Wei-Chia Chang · Chun-Ting Hou'],25,notes='Open with the release engineer deciding which completed patch needs extra inspection. This is task-end screening, not early prediction. Repository: '+team['repository_url'])
add('Which completed patches need extra review?',[
    'User: release engineer reviewing coding-agent patches',
    'Alert: open the execution trace and run targeted tests',
    'Normal = officially resolved; failure = not resolved',
    'A low score does not authorize an automatic release'],40,notes='The detector is an experimental triage signal. Failure labels do not establish deceptive intent or an explicit false completion claim. Source: SWE-bench ICLR 2024, https://proceedings.iclr.cc/paper_files/paper/2024/hash/edac78c3e300629acfe6cbe9ca88fb84-Abstract-Conference.html')
add('Failures are common in this dataset',[
    '284 real runs: 54 resolved / 230 unresolved',
    'Category 4: Text / Logs → 24 aggregate features',
    'Test: 150 failures / 162 runs (92.6%)',
    'High AP needs a prevalence baseline (0.926)',
    'No false-completion intent is inferred from the label'],40,notes='Of 300 Lite tasks, 16 have no generation; 284 generated trajectories are retained. 190 voluntary submissions include 140 failures; 94 budget-terminated submissions include 90 failures. Source: Dataset/evaluation_results.json and provenance.json. The application deviates from classical rare-event AD and this is a limitation.')
add('Training logs suggest workflow signals',[
    'EDA: training rows only (32 successes + 50 failures)',
    'Tokens received: W = 0.957; KS = 0.505',
    'Costs, steps and tokens overlap and correlate',
    'Hypothesis: workflow features improve ranking'],40,'EDA/figures/fig1_distribution_contrast.png',notes='W is standardized Wasserstein distance. Error patterns can be expected reproduction attempts; do not equate observed tracebacks with final failure. Summary statistics and Spearman matrix are computed only on training rows. Source: EDA/tables/feature_separability.csv.')
add('Normal-only reconstruction gives an alert score',[
    'Fit scaler and network on 32 successful training runs',
    '24 → 16 → 6 → 16 → 24; LayerNorm + ReLU',
    'Score = mean standardized squared reconstruction error',
    'Validation F2 cutoff: τ = 0.1758 for seed 42',
    's ≥ τ requests human review'],50,'Model/figures/fig0_pipeline.png',notes='Explain the six stages. AdamW lr .005, weight decay 1e-4, 150 epochs, batch 16. F2 weights false negatives more but is not the dollar-optimal objective. All features require the completed trajectory and final patch. Source: Model/models.py and Model/tables/all_seed_runs.csv.')
add('Three tiers share disjoint fitting and calibration data',[
    'Train 82 (32 normal / 50 failures) | Val 40 (10 / 30)',
    'Test 162 (12 / 150), same IDs for every tier',
    'T1: fixed error/repetition heuristic',
    'T2: normal-only autoencoder | T3: supervised XGBoost',
    'Five seeds; CPU; test was inspected in an earlier draft'],40,notes='The original test IDs were preserved during repairs. Fifty development failures moved to training. Validation never enters weight fitting. Test reuse is disclosed; these results require new runs for confirmation. No architecture search was performed. XGBoost defaults are deterministic across seeds. Sources: split_ids.json, environment.json. XGBoost: https://doi.org/10.1145/2939672.2939785')
values=[['Method','AP','ROC-AUC','F2','ms/sample']]
for i,x in r.iterrows(): values.append([['T1 heuristic','T2 autoencoder','T3 XGBoost'][i],f'{x.PR_AUC_mean:.3f} ± {x.PR_AUC_std:.3f}',f'{x.ROC_AUC_mean:.3f}',f'{x.F2_mean:.3f} ± {x.F2_std:.3f}',f'{x.Latency_ms_mean:.4f}'])
values.append(['Always alert','0.926','0.500','0.984','—'])
add('High AP does not establish a useful release gate',[
    'T2 vs T3: paired seed p = 0.449; no equivalence claim',
    'Mean T2: 142.4 TP / 7.6 FN / 9 FP / 3 TN',
    '75% false-alarm rate on only 12 successful test runs',
    'Always alert has higher F2; T2 spares some reviews'],55,table=values,notes='Timing is median warmed batch scoring divided by 162, not single-request end-to-end latency. Train seconds: T1 essentially zero, T2 .641 ± .866, T3 .013 ± .001. Paired row bootstrap AP gap T2 minus T3: [-.0147,.0206], seed 42 model, 1000 resamples. Seeds characterize optimization variation on one split. Source: main_results_table.csv, significance_tests.csv, practical_impact.csv.')
add('Ablations isolate three different decisions',[
    'H1: full AP 0.9783 vs patch-only 0.9639 (+0.0143)',
    'Patch-only F2 is slightly higher: no universal dominance',
    'H2: 3/32 contaminated rows lower AP by 0.0099',
    'H3: F1 and F2 choose identical cutoffs in every seed'],55,'Model/figures/fig5_ablation_curves.png',notes='Feature masks preserve all 24 coordinates and network shape. Contamination replaces 0,1,2,3 rows for 0,3.125,6.25,9.375%, holding sample size and clean scaler fixed. The 9.375% AP drop exceeds baseline seed SD .0039 by 2.5x. A Normal95 validation threshold misses 144 of 150 failures with zero false alarms. Sources: ablation_runs.csv and summary tables. H1 ranking supported; H2 supported here; H3 not supported.')
add('Noisy successes and quiet failures overlap',[
    'FP: scikit-learn-14894 | 25 steps, 19 errors | s=1.5674',
    'FN: django-14155 | 9 steps, 0 errors | s=0.1694',
    'FN: sympy-15345 | 11 steps, 0 errors | s=0.0630',
    'τ=0.1758; semantic correctness is missing from counts'],50,'Model/figures/fig6_feature_attribution.png',notes='Full IDs: scikit-learn__scikit-learn-14894, django__django-14155, sympy__sympy-15345. Scores minus tau: +1.3916,-.0065,-.1128. Patches add 8,15,2 lines respectively and all end in submit; do not claim aborts or empty patches. The separate flagged django__django-11630 has s=3.8673 and 14 stalled observations contributing 85.04% of reconstruction error; this supports trace inspection, not a causal verdict. Sources: failure_cases.csv, case_excerpts.json, feature_attribution_flagged_sample.csv.')
add('Under these assumed costs, reviewing all wins',[
    'Illustration: 500 tasks; 80% fail; $1,200 per escape',
    'Every alert costs 0.5 h × $60/h (including true alerts)',
    'T1: 114.7 escapes, 327.0 reviews → $147,410',
    'T2: 20.3 escapes, 454.7 reviews → $37,962',
    'Review all: $15,000, assuming perfect review'],45,notes='These are conditional projections, not measured production savings. T2 consumes 227.4 review hours vs T1 163.5. Perfect review is a strong scenario assumption; real reviewer sensitivity and capacity are unknown. At low deployment prevalence, precision changes. Source: practical_impact.csv. Contrast with prior work: SWE-agent solves issues; Deep SVDD uses one-class compactness; our work audits task-end trace screening. Sources: https://proceedings.neurips.cc/paper_files/paper/2024/hash/5a7c947568c1b1328ccc5230172e1e7c-Abstract-Conference.html and https://proceedings.mlr.press/v80/ruff18a.html')
add('Workflow signals help ranking; deployment remains unproven',[
    'Next: expected-error recovery and task-semantic checks',
    'Fresh agents/repos and more successes are needed',
    'No prefix prediction, live gate or production validation',
    'Member contributions: pending confirmation (all five)',
    'Codex audited code, experiments, references and artifacts'],40,notes='Team G04: '+', '.join(m['name']+' ('+m['student_id']+')' for m in team['members'])+'. Each member contribution is pending confirmation at the user’s request. Earlier materials mention Claude/Gemini but versions and scope remain unconfirmed. Do not imply team verification has already occurred. Full reproduction and source links are in README. Close with the research question: full workflow features improve AP over patch-only, but the alert trade-off is not a proven operational win.')
assert sum(s['seconds'] for s in slides)==480
Path('Presentation/content.json').write_text(json.dumps(slides,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
Path('Presentation/speaker_notes.md').write_text('# Eight-minute rehearsal guide\n\n'+ '\n\n'.join(f"## {i+1}. {s['title']} ({s['seconds']} s)\n\n{s['notes']}" for i,s in enumerate(slides))+'\n',encoding='utf-8')
print('11 slides, planned 480 seconds; actual rehearsal is still required.')
