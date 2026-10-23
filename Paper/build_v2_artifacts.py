"""Build v2 publication tables, final-size figures, and slide content from saved evidence."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

T = Path('Model/tables/v2')
G = Path('Paper/generated')
F = Path('Paper/figures/v2')
F.mkdir(parents=True, exist_ok=True)
G.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 8.5,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

LABELS = {
    'always_alert': 'Always alert (context)',
    'never_alert': 'Never alert (context)',
    'heuristic_T1': 'T1: Rule heuristic',
    'knn_R3_T2': 'T2: Dual encoder kNN',
    'supervised_R3_logistic_T3': 'T3: Encoder logistic',
    'supervised_R1_logistic': 'R1: TF-IDF logistic',
    'inductive_ecod_R0_clean': 'Inductive ECOD (R0-clean)',
    'isolation_forest_R0_clean': 'Isolation Forest (R0-clean)',
    'conditional_residual_R0_clean': 'Conditional residual (sec.)',
    'shallow_dae_R0_clean': 'Denoising AE (sec.)',
    'legacy_ae_R0': 'Numeric AE (v1 control)',
    'legacy_xgboost_R0': 'XGBoost (v1 control)',
}


def save(fig, name):
    fig.savefig(F / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(F / f'{name}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    df = pd.read_csv(T / 'main_results.csv')
    oof = pd.read_csv('Model/outputs/v2/oof_predictions.csv')
    folds = json.loads(Path('Dataset/v2/splits.json').read_text())['outer_folds']
    indexed = oof.set_index('instance_id')
    lat_map = json.loads((T / 'method_latencies.json').read_text())
    diag = json.loads((T / 'mechanistic_diagnostics.json').read_text())
    conf_tr = pd.read_csv(T / 'confirmatory_claude3opus_transfer.csv')
    conf_oof = pd.read_csv(T / 'confirmatory_claude3opus_oof.csv').set_index('model')

    # Compute per-fold metrics
    foldrows = []
    for r in df.itertuples():
        prefix = f'{r.model}_seed_{r.seed}'
        for fold in folds:
            z = indexed.loc[fold['test_instance_ids']]
            y = z.true_label
            s = z[prefix + '_score']
            pred = z[prefix + '_pred']
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            foldrows.append(dict(
                model=r.model,
                seed=r.seed,
                fold=fold['fold'],
                tn=tn, fp=fp, fn=fn, tp=tp,
                fpr=fp / (fp + tn),
                failure_recall=tp / (tp + fn),
                balanced_accuracy=0.5 * (tn / (tn + fp) + tp / (tp + fn)),
                roc_auc=roc_auc_score(y, s),
                average_precision=average_precision_score(y, s),
            ))
    fr = pd.DataFrame(foldrows)
    fr.to_csv(T / 'publication_fold_metrics.csv', index=False)

    f42_t2 = fr[(fr.model == 'knn_R3_T2') & (fr.seed == 42)].sort_values('fold')

    # 1. Table 2: Main Results Table (v2_main.tex)
    main_rows = []
    for model in LABELS:
        r = df[(df.model == model) & (df.seed == 42)].iloc[0]
        f = fr[(fr.model == model) & (fr.seed == 42)].sort_values('fold')
        z_seeds = df[df.model == model]
        seed_str = '' if len(z_seeds) == 1 else f' ({z_seeds.balanced_accuracy.std(ddof=1):.3f})'
        if model in ['always_alert', 'never_alert', 'knn_R3_T2']:
            p_str = '--'
        else:
            p_val = stats.ttest_rel(f.balanced_accuracy, f42_t2.balanced_accuracy).pvalue
            p_str = f'{p_val:.3f}'
        lat = lat_map[model]
        if model in ['always_alert', 'never_alert']:
            lat_str, tr_str = '--', '--'
        elif lat['embed_ms'] > 0:
            lat_str = f"{lat['infer_ms']:.2f}+3151"
            tr_str = f"{lat['train_s']:.3f}"
        else:
            lat_str = f"{lat['infer_ms']:.2f}"
            tr_str = f"{lat['train_s']:.3f}"

        if model in conf_oof.index:
            tr_sub = conf_tr[conf_tr.model == model]
            c_str = f"{tr_sub.balanced_accuracy.mean():.3f} / {conf_oof.loc[model, 'balanced_accuracy']:.3f}"
        else:
            c_str = '--'

        ba_str = f"\\textbf{{{r.balanced_accuracy:.3f}}}" if model == 'conditional_residual_R0_clean' else f"{r.balanced_accuracy:.3f}"
        fpr_str = f"\\textbf{{{r.fpr:.3f}}}" if model == 'inductive_ecod_R0_clean' else f"{r.fpr:.3f}"
        rec_str = f"\\textbf{{{r.failure_recall:.3f}}}" if model == 'supervised_R1_logistic' else f"{r.failure_recall:.3f}"
        ap_mean = f.average_precision.mean()
        ap_sd = f.average_precision.std(ddof=1)
        ap_str = f"\\textbf{{{ap_mean:.3f}}}$\\pm${ap_sd:.2f}" if model == 'supervised_R1_logistic' else f"{ap_mean:.3f}$\\pm${ap_sd:.2f}"

        main_rows.append(
            f"{LABELS[model]} & {ba_str} & {f.balanced_accuracy.mean():.3f}$\\pm${f.balanced_accuracy.std(ddof=1):.2f}{seed_str} & "
            f"{fpr_str} & {rec_str} & {int(r.tn)}/{int(r.fp)}/{int(r.fn)}/{int(r.tp)} & "
            f"{ap_str} & {p_str} & {lat_str} & {tr_str} & {c_str} \\\\"
        )
    (G / 'v2_main.tex').write_text('\n'.join(main_rows) + '\n')

    # Keep v2_variation.tex for backwards compatibility if referenced
    var_rows = []
    for model in ['knn_R3_T2', 'isolation_forest_R0_clean', 'legacy_ae_R0', 'shallow_dae_R0_clean']:
        z = df[df.model == model]
        ff = fr[(fr.model == model) & (fr.seed == 42)]
        seedstd = '--' if len(z) == 1 else f'{z.balanced_accuracy.std(ddof=1):.3f}'
        var_rows.append(f'{LABELS[model]} & {ff.balanced_accuracy.mean():.3f} $\\pm$ {ff.balanced_accuracy.std(ddof=1):.3f} & {seedstd} \\\\')
    (G / 'v2_variation.tex').write_text('\n'.join(var_rows) + '\n')

    # 2. Table 3: Representation & Channel Ablation (v2_ablation.tex)
    ab_stats = pd.read_csv(T / 'ablation_rep_channel_stats.csv').set_index('variant')
    ab_names = [
        ('R0_clean_euclidean', 'R0-clean num.'),
        ('R1_tfidf_cosine', 'R1 char TF-IDF'),
        ('R2_minilm_cosine', 'R2 MiniLM log'),
        ('patch_only_codebert', 'CodeBERT patch'),
        ('R3_dual_distance', 'R3 dual (T2)'),
    ]
    ab_rows = []
    for key, label in ab_names:
        r = ab_stats.loc[key]
        delta_str = '--' if key == 'R3_dual_distance' else f"{r.delta_ba_vs_r3:+.3f} [{r.boot_ci_low:+.2f},{r.boot_ci_high:+.2f}]"
        p_str = '--' if key == 'R3_dual_distance' else f"{r.paired_t_p:.3f}"
        ba_s = f"\\textbf{{{r.pooled_ba:.3f}}} ({r.fold_ba_mean:.2f}$\\pm${r.fold_ba_sd:.2f})" if key == 'R0_clean_euclidean' else f"{r.pooled_ba:.3f} ({r.fold_ba_mean:.2f}$\\pm${r.fold_ba_sd:.2f})"
        ab_rows.append(
            f"{label} & {ba_s} & {delta_str} & {p_str} & {r.fpr:.2f}/{r.recall:.2f} \\\\"
        )
    (G / 'v2_ablation.tex').write_text('\n'.join(ab_rows) + '\n')

    # 3. Table 4: Contamination Ablation (v2_contamination.tex)
    con = pd.read_csv(T / 'ablation_contamination.csv')
    con_ecod = pd.read_csv(T / 'ablation_contamination_ecod.csv')
    cg = con.groupby(['p_contamination', 'threshold_mode']).agg(
        ba_mean=('balanced_accuracy', 'mean'),
        ba_sd=('balanced_accuracy', 'std'),
        fpr_mean=('fpr', 'mean'),
        rec_mean=('failure_recall', 'mean'),
    ).reset_index()
    con_rows = []
    base_ba = cg[(cg.p_contamination == 0) & (cg.threshold_mode == 'clean_tau')].ba_mean.iloc[0]
    for r in cg.itertuples():
        p_pct = int(round(r.p_contamination * 100))
        actual_str = '0/32 (0\\%)' if p_pct == 0 else ('1/32 (3.1\\%)' if p_pct == 5 else '3/32 (9.4\\%)')
        mode_str = 'Clean $\\tau$' if r.threshold_mode == 'clean_tau' else 'Recal. $\\tau$'
        d_ba = r.ba_mean - base_ba
        con_rows.append(
            f"T2 ({mode_str}) & {actual_str} & {r.ba_mean:.3f}$\\pm${r.ba_sd:.3f} & {d_ba:+.3f} & {r.fpr_mean:.2f}/{r.rec_mean:.2f} \\\\"
        )
    ec_g = con_ecod.groupby('p_contamination').agg(
        ba_mean=('balanced_accuracy', 'mean'),
        ba_sd=('balanced_accuracy', 'std'),
        fpr_mean=('fpr', 'mean'),
        rec_mean=('failure_recall', 'mean'),
    ).reset_index()
    ec_base = ec_g[ec_g.p_contamination == 0].ba_mean.iloc[0]
    for r in ec_g.itertuples():
        p_pct = int(round(r.p_contamination * 100))
        actual_str = '0/32 (0\\%)' if p_pct == 0 else ('1/32 (3.1\\%)' if p_pct == 5 else '3/32 (9.4\\%)')
        d_ba = r.ba_mean - ec_base
        con_rows.append(
            f"ECOD (Recal. $\\tau$) & {actual_str} & {r.ba_mean:.3f}$\\pm${r.ba_sd:.3f} & {d_ba:+.3f} & {r.fpr_mean:.2f}/{r.rec_mean:.2f} \\\\"
        )
    (G / 'v2_contamination.tex').write_text('\n'.join(con_rows) + '\n')

    # 4. Table 5: Threshold Policy & Cost Ablation (v2_thresholds.tex)
    th_df = pd.read_csv(T / 'ablation_thresholds.csv')
    rule_labels = {
        'fpr_le_20': 'FPR$\\le$20\\%',
        'max_ba': 'Max val BA',
        'max_f1': 'Max val $F_1$',
        'max_f2': 'Max $F_2$/Min\\$',
    }
    th_rows = []
    for model_key, m_short in [('knn_R3_T2', 'T2'), ('inductive_ecod_R0_clean', 'ECOD')]:
        for rule_key, r_lab in rule_labels.items():
            r = th_df[(th_df.model == model_key) & (th_df.rule == rule_key)].iloc[0]
            tau_s = f"{r.mean_tau:.2f}" if r.mean_tau >= 1.0 else f"{r.mean_tau:.3f}"
            th_rows.append(
                f"{m_short} ({r_lab}) & {tau_s} & {r.balanced_accuracy:.3f} & {r.fpr:.2f}/{r.failure_recall:.2f} & "
                f"{r.f2:.3f} & \\${r.monthly_cost_usd_500:,.0f} \\\\"
            )
    th_rows.append("Always alert (Bayes) & $-\\infty$ & 0.500 & 1.00/\\textbf{1.00} & \\textbf{0.955} & \\textbf{\\$15,000} \\\\")
    (G / 'v2_thresholds.tex').write_text('\n'.join(th_rows) + '\n')

    # 5. Table 1: EDA on Fold 0 training partition (v2_stats.tex)
    raw = pd.read_csv('Dataset/sweagent_gpt4_features.csv').set_index('instance_id')
    train = raw.loc[folds[0]['train_instance_ids']]
    n0 = train[train.label == 0]
    f0 = train[train.label == 1]
    stats_rows = []
    eda_cols = [
        ('tokens_received', 'Received tokens'),
        ('total_cost_usd', 'API cost (USD)'),
        ('num_steps', 'Trajectory steps'),
        ('num_error_observations', 'Error observations'),
        ('patch_char_length', 'Patch chars'),
        ('patch_lines_added', 'Patch lines added'),
    ]
    for c, label in eda_cols:
        v_all = train[c].to_numpy(float)
        std_all = (v_all - v_all.mean()) / (v_all.std() + 1e-8)
        w = stats.wasserstein_distance(std_all[train.label == 0], std_all[train.label == 1])
        ks, p = stats.ks_2samp(n0[c], f0[c])
        iqr_n = np.percentile(n0[c], 75) - np.percentile(n0[c], 25)
        iqr_f = np.percentile(f0[c], 75) - np.percentile(f0[c], 25)
        if c == 'total_cost_usd':
            n_s = f"{n0[c].median():.2f} ({iqr_n:.2f})"
            f_s = f"{f0[c].median():.2f} ({iqr_f:.2f})"
        else:
            n_s = f"{n0[c].median():.0f} ({iqr_n:.0f})"
            f_s = f"{f0[c].median():.0f} ({iqr_f:.0f})"
        p_s = f"{p:.1e}" if p < 0.001 else f"{p:.3f}"
        stats_rows.append(f"{label} & {n_s} & {f_s} & {w:.3f} & {ks:.3f} & {p_s} \\\\")
    (G / 'v2_stats.tex').write_text('\n'.join(stats_rows) + '\n')

    # 6. Table 6: Error Case Studies Table (v2_cases.tex)
    ec = diag['enriched_cases']
    case_rows = []
    for key, tag in [('highest_fp', 'FP'), ('lowest_fn', 'FN'), ('borderline_error', 'FP*')]:
        c = ec[key]
        iid_short = c['instance_id'].split('__')[-1]
        case_rows.append(
            f"\\texttt{{{iid_short}}} & {tag} & {c['outer_fold']} ({c['same_repo_normal_ref_count']}/32) & "
            f"{c['num_steps']}/{c['num_error_observations']}/{c['patch_lines_added']} & "
            f"{c['log_distance_to_nearest']:.3f}/{c['patch_distance_to_nearest']:.3f} & "
            f"{c['score']:.3f} & {c['tau']:.3f} ({c['margin_to_tau']:+.4f}) \\\\"
        )
    (G / 'v2_cases.tex').write_text('\n'.join(case_rows) + '\n')

    # =========================================================================
    # FIGURES
    # =========================================================================
    # Figure 1: EDA Boxplots (eda.pdf)
    fig, ax = plt.subplots(1, 2, figsize=(3.30, 1.38))
    for a, c, label, unit in zip(
        ax,
        ['tokens_received', 'patch_char_length'],
        ['Received tokens ($W$=0.999)', 'Patch length ($W$=0.203)'],
        ['$\\log(1 + \\mathrm{tokens})$', '$\\log(1 + \\mathrm{chars})$'],
    ):
        bp = a.boxplot(
            [np.log1p(train.loc[train.label == y, c]) for y in [0, 1]],
            tick_labels=['Success\n(n=32)', 'Failure\n(n=138)'],
            showfliers=False,
            widths=0.52,
            patch_artist=True,
        )
        for patch, color in zip(bp['boxes'], ['#A9CBE8', '#F4B6A8']):
            patch.set_facecolor(color)
            patch.set_edgecolor('#15283B')
        for median in bp['medians']:
            median.set_color('#15283B')
            median.set_linewidth(1.5)
        a.set_title(label, fontsize=8.0)
        a.tick_params(labelsize=7.0)
        a.set_ylabel(unit, fontsize=7.0)
    fig.tight_layout(pad=0.3)
    save(fig, 'eda')

    # Figure 2: 6-Checkpoint End-to-End Pipeline Figure (pipeline.pdf)
    fig, ax = plt.subplots(figsize=(7.2, 1.05))
    ax.set_xlim(0, 12.1)
    ax.set_ylim(0, 1.75)
    ax.axis('off')
    stages = [
        ("1. Inputs & Hygiene", "Action/obs logs + diff\nStrip ANSI & <WORKDIR>\nDrop gold/eval/IDs"),
        ("2. Frozen Encoding", "MiniLM: 224-tok chunks\nCodeBERT: 480-tok hunks\nL2-norm 384D + 768D"),
        ("3. Normal Reference", "Store only resolved\ntraining runs (y=0)\nn_ref = 32-33 / fold"),
        ("4. Dual kNN Score", "d(x,r)=(d_log+d_patch)/2\ns(x) = mean k-NN dist\nk in {1,3,5}, s in [0,1]"),
        ("5. Val Calibration", "Tune k, tau on val only\nMax recall s.t. FPR<=20%\nFreeze (k*, tau*)"),
        ("6. Triage & Output", "Alert if s(x) >= tau*\nRoute to human review\nReport OOF & holdout"),
    ]
    colors = ['#EBF3FA', '#E1EEF8', '#D6E8F5', '#CBE2F3', '#C0DCF0', '#B5D6EE']
    for idx, (title, body) in enumerate(stages):
        x0 = 0.06 + idx * 2.01
        box = FancyBboxPatch(
            (x0, 0.10), 1.84, 1.54,
            boxstyle="round,pad=0.05,rounding_size=0.09",
            facecolor=colors[idx], edgecolor='#1B3B5F', linewidth=1.0
        )
        ax.add_patch(box)
        ax.text(x0 + 0.92, 1.40, title, ha='center', va='center', fontsize=6.5, fontweight='bold', color='#0E2439')
        ax.text(x0 + 0.92, 0.68, body, ha='center', va='center', fontsize=5.7, color='#15283B', linespacing=1.22)
        if idx < 5:
            ax.annotate(
                '', xy=(x0 + 2.00, 0.87), xytext=(x0 + 1.88, 0.87),
                arrowprops=dict(arrowstyle='->', lw=1.2, color='#1B3B5F')
            )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    save(fig, 'pipeline')

    # Figure 3: 4-Panel Confusion Matrix Figure (confusion.pdf)
    fig, axs = plt.subplots(1, 4, figsize=(6.85, 1.42))
    cm_models = [
        ('knn_R3_T2', 'T2: Dual encoder\n(BA 0.481, FPR 0.259)'),
        ('supervised_R3_logistic_T3', 'T3: Encoder logistic\n(BA 0.596, FPR 0.352)'),
        ('inductive_ecod_R0_clean', 'Inductive ECOD\n(BA 0.657, FPR 0.130)'),
        ('conditional_residual_R0_clean', 'Conditional residual\n(BA 0.675, FPR 0.185)'),
    ]
    for idx, (ax, (model, title)) in enumerate(zip(axs, cm_models)):
        r = df[(df.model == model) & (df.seed == 42)].iloc[0]
        cm = np.array([[r.tn, r.fp], [r.fn, r.tp]], int)
        frac = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(frac, vmin=0, vmax=1, cmap='Blues')
        for i in range(2):
            for j in range(2):
                pct = 100.0 * frac[i, j]
                ax.text(
                    j, i, f"{cm[i, j]}\n({pct:.0f}%)",
                    ha='center', va='center', fontsize=7.8, fontweight='bold',
                    color='white' if frac[i, j] > 0.5 else '#15283B'
                )
        ax.set_xticks([0, 1], ['No alert', 'Alert'])
        ax.set_yticks([0, 1], ['Success\n(n=54)', 'Failure\n(n=230)'] if idx == 0 else ['', ''])
        ax.tick_params(labelsize=7.0)
        ax.set_title(title, fontsize=7.6)
        ax.set_xlabel('Predicted decision', fontsize=7.0)
        if idx == 0:
            ax.set_ylabel('Actual class', fontsize=7.0)
    fig.tight_layout(pad=0.25)
    save(fig, 'confusion')

    # Figure 4: 2-Panel Ablation Figure (ablations.pdf)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.85, 1.52))
    rep_order = ['R0_clean_euclidean', 'patch_only_codebert', 'R3_dual_distance', 'R2_minilm_cosine', 'R1_tfidf_cosine']
    rep_short = ['R0-clean numeric', 'CodeBERT patch', 'R3 dual (T2)', 'R2 MiniLM log', 'R1 char TF-IDF']
    means = [ab_stats.loc[k, 'fold_ba_mean'] for k in rep_order]
    sds = [ab_stats.loc[k, 'fold_ba_sd'] for k in rep_order]
    bar_colors = ['#2B5C8F', '#4A89C8', '#0E2439', '#7BAFD4', '#A3C1AD']
    y_pos = np.arange(len(rep_order))
    ax1.barh(y_pos, means, xerr=sds, capsize=2.5, color=bar_colors, height=0.62, edgecolor='#15283B', linewidth=0.6)
    ax1.axvline(0.50, ls='--', c='#8B0000', lw=0.9, label='Random (0.50)')
    ax1.set_yticks(y_pos, rep_short, fontsize=7.4)
    ax1.set_xlim(0.25, 0.78)
    ax1.set_xlabel('5-fold Balanced Accuracy (mean $\\pm$ SD)', fontsize=8.0)
    ax1.set_title('(a) kNN Representation & Channel Ablation', fontsize=8.5)
    ax1.invert_yaxis()
    ax1.legend(fontsize=7.0, loc='lower right')

    for mode, label, color, marker in [
        ('clean_tau', 'T2: Fixed clean $\\tau$', '#0E2439', 'o'),
        ('recalibrated_tau', 'T2: Recalibrated $\\tau$', '#4A89C8', 's'),
    ]:
        z = con[con.threshold_mode == mode].groupby('p_contamination').balanced_accuracy.agg(['mean', 'std'])
        ax2.errorbar(z.index * 100, z['mean'], yerr=z['std'], marker=marker, color=color, capsize=3, lw=1.3, label=label)
    z_ec = con_ecod.groupby('p_contamination').balanced_accuracy.agg(['mean', 'std'])
    ax2.errorbar(z_ec.index * 100, z_ec['mean'], yerr=z_ec['std'], marker='^', color='#2E7D32', capsize=3, lw=1.3, label='Inductive ECOD (recal. $\\tau$)')
    ax2.axhline(0.50, ls='--', c='#8B0000', lw=0.8)
    ax2.set_xticks([0, 5, 10], ['0% (0/32)', '5% (1/32)', '10% (3/32)'])
    ax2.set_ylim(0.44, 0.70)
    ax2.set_xlabel('Nominal (actual) normal-reference contamination', fontsize=8.0)
    ax2.set_ylabel('Balanced Accuracy (mean $\\pm$ SD)', fontsize=8.0)
    ax2.set_title('(b) AD-Specific Reference Contamination', fontsize=8.5)
    ax2.legend(fontsize=6.8, loc='center right')
    fig.tight_layout(pad=0.45)
    save(fig, 'ablations')

    # Also save standalone representations.pdf and contamination.pdf for slides
    fig_r, ax_r = plt.subplots(figsize=(3.35, 1.9))
    ax_r.barh(y_pos, means, xerr=sds, capsize=2.5, color=bar_colors, height=0.62)
    ax_r.axvline(0.50, ls='--', c='black', lw=0.8)
    ax_r.set_yticks(y_pos, [s.replace('\n', ' ') for s in rep_short], fontsize=7.5)
    ax_r.set_xlim(0.25, 0.78)
    ax_r.set_xlabel('Balanced accuracy (mean $\\pm$ SD)')
    ax_r.invert_yaxis()
    fig_r.tight_layout(pad=0.4)
    save(fig_r, 'representations')

    fig_c, ax_c = plt.subplots(figsize=(3.35, 1.9))
    for mode, label in [('clean_tau', 'T2 fixed clean $\\tau$'), ('recalibrated_tau', 'T2 recalibrated $\\tau$')]:
        z = con[con.threshold_mode == mode].groupby('p_contamination').balanced_accuracy.agg(['mean', 'std'])
        ax_c.errorbar(z.index * 100, z['mean'], yerr=z['std'], marker='o', capsize=3, label=label)
    ax_c.set_xlabel('Requested replacement (%)')
    ax_c.set_ylabel('Balanced accuracy')
    ax_c.legend(fontsize=7)
    fig_c.tight_layout(pad=0.4)
    save(fig_c, 'contamination')

    # Figure 5: 2-Panel Interpretability & Attribution Figure (interpretability.pdf)
    step_dict = json.loads(Path('Dataset/v2/embeddings/step_embeddings.json').read_text())
    r2_npz = np.load('Dataset/v2/embeddings/r2_minilm.npz')
    r2_map = {str(x): r2_npz['embeddings'][i] for i, x in enumerate(r2_npz['instance_ids'])}
    from tools.build_text_embeddings import compute_channel_distance

    fp_steps = [np.array(v, dtype=np.float32) for v in step_dict['django__django-13230']]
    fp_ref = r2_map['django__django-11133']
    fp_dists = [compute_channel_distance(v, fp_ref) for v in fp_steps]

    fn_steps = [np.array(v, dtype=np.float32) for v in step_dict['sympy__sympy-12454']]
    fn_ref = r2_map['sympy__sympy-13647']
    fn_dists = [compute_channel_distance(v, fn_ref) for v in fn_steps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.85, 1.50))
    ax1.plot(range(1, len(fp_dists) + 1), fp_dists, marker='o', ms=3.5, lw=1.3, color='#C0392B', label='FP django-13230 ($d_{\\mathrm{log}}$=0.203)')
    ax1.plot(range(1, len(fn_dists) + 1), fn_dists, marker='s', ms=3.5, lw=1.3, color='#1F618D', label='FN sympy-12454 ($d_{\\mathrm{log}}$=0.047)')
    ax1.set_xlabel('Trajectory step index $t$', fontsize=8.0)
    ax1.set_ylabel('Step cosine distance $(1-\\cos)/2$', fontsize=8.0)
    ax1.set_title('(a) T2 Step-to-Reference Log Distance Profile', fontsize=8.5)
    ax1.legend(fontsize=7.0, loc='lower right')

    top_f = diag['ecod_flagged_attribution']['top_features'][:6]
    other_pct = 100.0 - sum(x['pct'] for x in top_f)
    f_labels = [
        'patch_char_length (2,310)',
        'patch_lines_added (64)',
        'patch_files_changed (3)',
        'patch_edit_ratio (1.00)',
        'repeated_tool_streak (15)',
        'stalled_turns (14)',
        'Other 17 features',
    ]
    f_pcts = [x['pct'] for x in top_f] + [other_pct]
    y_f = np.arange(len(f_labels))
    f_colors = ['#1B4F72'] * 4 + ['#2874A6'] * 2 + ['#85929E']
    ax2.barh(y_f, f_pcts, color=f_colors, height=0.62, edgecolor='#15283B', linewidth=0.5)
    for idx_f, val_p in enumerate(f_pcts):
        ax2.text(val_p + 0.8, idx_f, f'{val_p:.1f}%', va='center', fontsize=6.8)
    ax2.set_yticks(y_f, f_labels, fontsize=7.2)
    ax2.set_xlim(0, 62)
    ax2.set_xlabel('Share of total ECOD score $s(x)=45.21$ (%)', fontsize=8.0)
    ax2.set_title('(b) Additive Tail Attribution (django-11630, TP)', fontsize=8.5)
    ax2.invert_yaxis()
    fig.tight_layout(pad=0.45)
    save(fig, 'interpretability')

    # Reusable content for the 11-slide / 480-second presentation builder
    slides = []
    def add(title, lines, sec, notes, image=None, table=None):
        slides.append(dict(title=title, lines=lines, seconds=sec, notes=notes, image=image, table=table))

    add('Can local text embeddings detect agent failures?', [
        'Team G04 · NYCU Data Mining HW2',
        '284 SWE-agent GPT-4 runs + 271 untouched Claude-3-Opus runs',
        'MiniLM logs + CodeBERT patches, computed locally',
        'Frozen encoders lose to simple numeric controls (BA 0.481 vs 0.657)',
    ], 30, 'Task-end failure screening on SWE-bench Lite. Zero new LLM API calls. Official execution verdicts define ground truth. Full author list and student IDs appear in the report.')
    add('The decision is whether a completed run needs review', [
        'Normal reference: 54 officially resolved tasks (19.0%)',
        'Failure label: 230 unresolved tasks (81.0%), including 17 silent failures (6.0%)',
        'Errors, large patches and high cost do not prove failure',
        'High failure prevalence limits classical rare-anomaly assumptions',
    ], 40, 'Category 4: Text / Logs. Task-end screening matches PR integration queues. We also isolate the rare 6.0% silent semantic failure subtype.', None)
    add('Training-only analysis shows overlapping behavior', [
        'EDA uses outer fold 0 training rows only (32 normal, 138 failure)',
        'Tokens received separates strongly (W = 0.999, KS = 0.520)',
        'Exact redundancy (Spearman 1.000) removes duplicate tool-call count',
        'Heavy right tails motivate log1p and normal-only median/IQR scaling',
    ], 35, 'No held-out row enters these boxplots or Wasserstein/KS statistics. Source: Paper/figures/v2/eda and v2_stats.tex.', image='Paper/figures/v2/eda.png')
    add('Six-stage primary pipeline with frozen encoders', [
        'MiniLM: 224-token log chunks, overlap 32, step + run mean pooling',
        'CodeBERT: 480-token diff hunks, overlap 64, token-weighted pooling',
        'Normal-only reference bank (~32 resolved runs per fold)',
        'T2 score: mean dual cosine distance to k in {1,3,5} nearest normals',
    ], 45, 'Encoders remain frozen. Validation selects k and threshold tau under empirical FPR <= 20% maximizing recall.', image='Paper/figures/v2/pipeline.png')
    add('Strict 5-fold CV plus untouched confirmatory holdout', [
        'Primary: 5 stratified outer folds (N=284, ~60/20/20% train/val/test)',
        'Confirmatory: 271 untouched SWE-agent Claude-3-Opus trajectories',
        'Both zero-shot cross-agent transfer and locked 5-fold replication',
        'Paired fold t-tests and 2,000-resample stratified paired bootstrap CIs',
    ], 45, 'Addresses prior test inspection on GPT-4 by evaluating frozen models once on the untouched 271 Claude-3-Opus holdout.')
    values = [['Method', 'GPT-4 BA', 'FPR', 'Recall', 'Opus Holdout BA']]
    for model, short_n in [
        ('heuristic_T1', 'T1: Heuristic'),
        ('knn_R3_T2', 'T2: Dual encoder'),
        ('supervised_R3_logistic_T3', 'T3: Supervised'),
        ('inductive_ecod_R0_clean', 'Inductive ECOD'),
        ('conditional_residual_R0_clean', 'Conditional res.'),
    ]:
        r = df[(df.model == model) & (df.seed == 42)].iloc[0]
        tr_m = conf_tr[conf_tr.model == model].balanced_accuracy.mean()
        oof_m = conf_oof.loc[model, 'balanced_accuracy']
        values.append([short_n, f'{r.balanced_accuracy:.3f}', f'{r.fpr:.3f}', f'{r.failure_recall:.3f}', f'{tr_m:.3f} / {oof_m:.3f}'])
    add('Numeric controls outperform the embedding detector', [
        'Seed 42 OOF across 284 GPT-4 runs and 271 untouched Opus runs',
        'ECOD beats T2 by +0.176 BA (p = 0.0037, 95% CI [+0.114, +0.240])',
        'Findings replicate on untouched Claude-3-Opus holdout (0.645 vs 0.542)',
    ], 55, 'All numbers derive from main_results.csv, bootstrap_ci.csv, and confirmatory_claude3opus_oof.csv.', table=values)
    add('The confusion matrices expose the missed failures', [
        'T2: 179 of 230 failures receive no alert (recall 22.2%)',
        'Only 40 of 219 unflagged T2 runs actually succeed (18.3%)',
        'Conditional residual catches 47.1% of rare silent failures vs 0% for T1',
    ], 50, 'T2 TN 40 FP 14 FN 179 TP 51; ECOD TN 47 FP 7 FN 128 TP 102; Conditional TN 44 FP 10 FN 107 TP 123.', image='Paper/figures/v2/confusion.png')
    add('Mechanistic ablation: log variance drowns out patch signal', [
        'Numeric kNN reaches 0.641 BA; CodeBERT patch-only reaches 0.594',
        'Dual encoder drops to 0.481; MiniLM log-only is 0.456',
        'CodeBERT anisotropy compresses patch distances (mean 0.009 vs log 0.143)',
        'Log distance drives 98.2% of unweighted fusion variance',
    ], 50, 'Checked directly in embedding distance data: d_log variance is 56x larger than d_patch variance.', image='Paper/figures/v2/representations.png')
    add('Contamination & threshold policy ablations', [
        'Clean failure and normal dual distances overlap (0.077 vs 0.073)',
        '10% contamination shifts failure scores by only -0.00014 (H3 refuted)',
        'F2 and $-cost rules collapse to flagging 95-100% of runs under 81% skew',
        'Always-alert ($15,000/mo) beats all detectors unless failure rate < 2.5%',
    ], 40, 'Threshold ablation table compares FPR<=20%, max-BA, max-F1, and max-F2/min-cost rules.', image='Paper/figures/v2/contamination.png')
    add('Error case studies and additive feature attribution', [
        'FP django-13230 (s=.1128 > tau=.1087): 9 errors inflate d_log (.2029)',
        'FN sympy-12454 (s=.0248 < tau=.0731): quiet 1-line edit looks normal',
        'Boundary FP sympy-24213 (margin +.00018): 11-normal val step noise',
        'ECOD additive tail attribution highlights patch size and stalled turns',
    ], 50, 'Full IDs, SHA-256s, raw feature values, and repository overlap counts are reported in Table 6 and Figure 5.', image='Paper/figures/v2/interpretability.png')
    add('Takeaways, deployment limits, and next steps', [
        'No method meets all four practical release gates on GPT-4 or Opus',
        'Variance-standardized channel weighting and test-execution verification next',
        'CPU extraction takes 3.15 s/run vs 0.012 ms/run for Inductive ECOD',
        'All five member contributions remain pending confirmation',
    ], 40, 'Gates: recall>=90%, FPR<=20%, unflagged success>=80%, coverage>=5%. AI assistance and all five author IDs appear in the report.')

    assert len(slides) == 11 and sum(x['seconds'] for x in slides) == 480
    Path('Presentation/content.json').write_text(json.dumps(slides, indent=2) + '\n')
    Path('Presentation/speaker_notes.md').write_text(
        '# V2 eight-minute rehearsal guide\n\n' +
        '\n\n'.join(f"## {i+1}. {s['title']} ({s['seconds']} seconds)\n\n{s['notes']}" for i, s in enumerate(slides)) + '\n'
    )
    print('Generated all v2 tables, figures, and 11-slide/480-second presentation content.')


if __name__ == '__main__':
    main()
