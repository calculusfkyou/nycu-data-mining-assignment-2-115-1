"""Generate professional oral presentation slides for HW2.

Adheres strictly to DM-HW2 Section 21:
- 8-minute presentation narrative
- 11 slides covering: Title, Motivation, Problem Formulation, Data & Insight,
  Method Pipeline, Challenges, 3-Tier Results, Ablations, Error Analysis,
  Impact & Related Work, Conclusion & Contributions.
- High-contrast, clean typography, 16:9 widescreen layout with embedded figures.
"""

from __future__ import annotations

import os
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt


OUTPUT_DIR = Path("Presentation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Nature/ACL inspired Color Palette
COLOR_NAVY = RGBColor(15, 77, 146)       # #0F4D92 Primary brand
COLOR_BLUE = RGBColor(55, 117, 186)     # #3775BA Secondary blue
COLOR_RED = RGBColor(182, 67, 66)       # #B64342 Anomaly / Alert
COLOR_TEAL = RGBColor(66, 148, 158)     # #42949E Accent teal
COLOR_DARK = RGBColor(26, 33, 31)       # #1A211F Body text
COLOR_MUTED = RGBColor(86, 101, 95)     # #56655F Subtitles / secondary
COLOR_CARD = RGBColor(246, 248, 247)    # #F6F8F7 Card background
COLOR_BORDER = RGBColor(214, 222, 218)  # #D6DEDA Card border
COLOR_WHITE = RGBColor(255, 255, 255)


def add_header(slide, title: str, category: str = "AI AGENT GHOSTWORK DETECTOR"):
    """Standardized slide header."""
    header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.1))
    tf = header_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_cat = tf.paragraphs[0]
    p_cat.text = category.upper()
    p_cat.font.size = Pt(10)
    p_cat.font.bold = True
    p_cat.font.color.rgb = COLOR_NAVY

    p_title = tf.add_paragraph()
    p_title.text = title
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_DARK
    p_title.space_before = Pt(3)


def add_card(slide, left, top, width, height, title: str = "", bg_color=COLOR_CARD, border_color=COLOR_BORDER):
    """Add a structured container card."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    card.line.color.rgb = border_color
    card.line.width = Pt(1)

    if title:
        tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.15), width - Inches(0.4), Inches(0.4))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = COLOR_NAVY
    return card


def build_presentation() -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # ==========================================================================
    # Slide 1: Title & Team
    # ==========================================================================
    s1 = prs.slides.add_slide(blank_layout)

    # Accent decorative bar
    bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.5), Inches(0.15), Inches(4.5))
    bar.fill.solid()
    bar.fill.fore_color.rgb = COLOR_NAVY
    bar.line.fill.background()

    tb = s1.shapes.add_textbox(Inches(1.2), Inches(1.5), Inches(11.0), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "DATA MINING · FALL 2026 · HW2 ANOMALY DETECTION"
    p0.font.size = Pt(12)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_NAVY

    p1 = tf.add_paragraph()
    p1.text = "AI Agent Ghostwork Detector"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_DARK
    p1.space_before = Pt(8)

    p2 = tf.add_paragraph()
    p2.text = "Forecasting Early Task Failures and False Completion Claims from SWE-bench Trajectories"
    p2.font.size = Pt(18)
    p2.font.color.rgb = COLOR_MUTED
    p2.space_before = Pt(6)

    p3 = tf.add_paragraph()
    p3.text = "Team G07: Dun-Jie Yang (Fergus4506) · Charlie Wang (calculusfkyou) · Mo-Ty Wang (MO7YW4NG) · Wei-Chia Huang (lasvigas0701)\nDepartment of Computer Science, National Yang Ming Chiao Tung University (NYCU)\nGitHub: https://github.com/calculusfkyou/nycu-data-mining-assignment-2-115-1"
    p3.font.size = Pt(11)
    p3.font.color.rgb = COLOR_DARK
    p3.space_before = Pt(28)

    # ==========================================================================
    # Slide 2: Motivation & Scenario
    # ==========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Motivation: The Threat of Silent Agent Ghostwork in CI/CD")

    # Card 1: The Problem
    add_card(s2, Inches(0.8), Inches(1.6), Inches(3.6), Inches(5.2), "The Silent Failure Problem")
    tb = s2.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(3.2), Inches(4.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "• AI coding agents operate multi-turn workflows (open, edit, run, bash) in complex codebases.\n\n• Critical Blind Spot: Agents frequently self-declare task completion with false confidence while the bug remains unverified.\n\n• In 190 voluntary submissions in SWE-bench Lite, 73.7% (140) are FALSE COMPLETIONS (Ghostwork)."
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_DARK

    # Card 2: Stakeholder & Action
    add_card(s2, Inches(4.7), Inches(1.6), Inches(3.6), Inches(5.2), "Stakeholders & Operational Action")
    tb = s2.shapes.add_textbox(Inches(4.9), Inches(2.2), Inches(3.2), Inches(4.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "• Target Users: CI/CD release engineers and agent platform operators.\n\n• Trigger Action: Score s(x) > τ triggers automated Hold-and-Quarantine.\n\n• Consequence: Halts automated PR merging; routes failed trajectory directly to human review, preventing broken releases."
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_DARK

    # Card 3: Research Question
    add_card(s2, Inches(8.6), Inches(1.6), Inches(3.9), Inches(5.2), "Core Research Question")
    tb = s2.shapes.add_textbox(Inches(8.8), Inches(2.2), Inches(3.5), Inches(4.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Research Question:\nCan we reliably forecast task failure before an agent submits, and does watching the full workflow process catch failures better than inspecting the final patch alone?\n\nKey Finding:\nWatching how the agent works (process dynamics) improves PR-AUC by +0.012 and F2 by +0.019 over patch diffs alone."
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_DARK

    # ==========================================================================
    # Slide 3: Problem Formulation & Anomaly Definition
    # ==========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Task Formulation: Ghostwork Definition & Extreme Imbalance")

    add_card(s3, Inches(0.8), Inches(1.6), Inches(5.6), Inches(2.5), "Normal Sample: Verified Resolution")
    tb = s3.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(5.2), Inches(1.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Instance: astropy__astropy-14182\n• Efficient 18-step trajectory, zero lingering tool errors.\n• Reproduces bug, modifies 1 target file with focused 12-line fix.\n• Result: Pass all regression test suites (Normal, y = 0)."
    p.font.size = Pt(10.5)
    p.font.color.rgb = COLOR_DARK

    add_card(s3, Inches(6.8), Inches(1.6), Inches(5.7), Inches(2.5), "Anomaly Sample: Agent Ghostwork")
    tb = s3.shapes.add_textbox(Inches(7.0), Inches(2.1), Inches(5.3), Inches(1.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Instance: astropy__astropy-14365\n• 36-step trajectory with 8 recurring error cascades.\n• Agent declares: 'Issue fixed and verified, submitting changes.'\n• Ground Truth: External test harness fails (Ghostwork Anomaly, y = 1)."
    p.font.size = Pt(10.5)
    p.font.color.rgb = COLOR_RED

    add_card(s3, Inches(0.8), Inches(4.4), Inches(11.7), Inches(2.4), "Category 4 (Text / Logs) & Class Imbalance Hygiene")
    tb = s3.shapes.add_textbox(Inches(1.0), Inches(4.9), Inches(11.3), Inches(1.7))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "• Data Category: Category 4 (Text / Logs) — Unstructured command logs, tool execution traces, and patch diffs.\n• Real Dataset: 284 SWE-agent GPT-4 trajectories on SWE-bench Lite (54 Resolved / 230 Failed, ratio 1:4.26).\n• Imbalance Failure Mode: Accuracy achieves 81.0% by trivially predicting 'always failure' (zero utility). We mandate PR-AUC, ROC-AUC, and cost-sensitive F2."
    p.font.size = Pt(10.5)
    p.font.color.rgb = COLOR_DARK

    # ==========================================================================
    # Slide 4: Data & Key Insight (EDA)
    # ==========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Exploratory Insight: Process Dynamics Dominate Static Patches")

    # Embed EDA Distribution Contrast image
    img_path = "EDA/figures/fig1_distribution_contrast.png"
    if os.path.exists(img_path):
        s4.shapes.add_picture(img_path, Inches(0.8), Inches(1.6), Inches(6.8), Inches(5.2))

    add_card(s4, Inches(7.8), Inches(1.6), Inches(4.7), Inches(5.2), "Key Statistical Insights (§4)")
    tb = s4.shapes.add_textbox(Inches(8.0), Inches(2.2), Inches(4.3), Inches(4.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "1. Process Dynamics Rank Highest:\n• Tokens Received: W = 0.910, KS = 0.440 (p = 2.4e-5)\n• Total Cost (USD): W = 0.828, KS = 0.383 (p = 3.8e-4)\n• Tool Execution Steps: W = 0.810, KS = 0.371\n\n2. Weakness of Static Patches:\n• Patch Char Length: W = 0.189 (KS p = 0.029)\n• Diff size alone does not indicate correctness!\n\n3. Hypotheses Established:\n• H1: Process features dominate patch diffs in PR-AUC.\n• H2: Contamination beyond 5% collapses autoencoders.\n• H3: F2 thresholding suppresses silent bug escapes."
    p.font.size = Pt(10.5)
    p.font.color.rgb = COLOR_DARK

    # ==========================================================================
    # Slide 5: Method Pipeline
    # ==========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Method: 6-Checkpoint Bottleneck Autoencoder Pipeline")

    # Embed Pipeline figure
    pipe_path = "Model/figures/fig0_pipeline.png"
    if os.path.exists(pipe_path):
        s5.shapes.add_picture(pipe_path, Inches(0.8), Inches(1.6), Inches(11.7), Inches(2.8))

    add_card(s5, Inches(0.8), Inches(4.6), Inches(5.6), Inches(2.2), "Anomaly Scoring Formulation s(x)")
    tb = s5.shapes.add_textbox(Inches(1.0), Inches(5.1), Inches(5.2), Inches(1.5))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "• Core Model: Symmetric Bottleneck Autoencoder (24 -> 16 -> 6 -> 16 -> 24) with LayerNorm + ReLU.\n• Training: Strictly on clean normal trajectories (N = 32).\n• Score: Mean squared reconstruction error:\n  s(x) = (1/D) * sum_{d=1}^D (x_d - x_hat_d)^2"
    p.font.size = Pt(10)
    p.font.color.rgb = COLOR_DARK

    add_card(s5, Inches(6.8), Inches(4.6), Inches(5.7), Inches(2.2), "Cost-Sensitive Threshold Decision (tau)")
    tb = s5.shapes.add_textbox(Inches(7.0), Inches(5.1), Inches(5.3), Inches(1.5))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "• Cost Asymmetry: FN (unverified commit escapes to prod) costs 40x more than FP (manual audit review).\n• Threshold Rule: Calibrated strictly on validation split to maximize F2 score (tau = 0.1182).\n• Operational Alert: Scores >= tau route to CI quarantine."
    p.font.size = Pt(10)
    p.font.color.rgb = COLOR_DARK

    # ==========================================================================
    # Slide 6: Challenges & Protocol Hygiene
    # ==========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Protocol Hygiene: Zero-Leakage & Operational Challenges")

    add_card(s6, Inches(0.8), Inches(1.6), Inches(3.6), Inches(5.2), "Partition Hygiene")
    tb = s6.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(3.2), Inches(4.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "• Normal-Only Training:\n  Tier 2 trained strictly on resolved workflows (N = 32).\n\n• Validation-Only Tuning:\n  tau tuned on validation split (N = 90); never used for gradient updates.\n\n• Frozen Holdout Test:\n  Identical test split (N = 162) evaluated once across all 3 tiers."
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_DARK

    add_card(s6, Inches(4.7), Inches(1.6), Inches(3.6), Inches(5.2), "Operational Challenges")
    tb = s6.shapes.add_textbox(Inches(4.9), Inches(2.2), Inches(3.2), Inches(4.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "• Ghostwork vs. Timeout:\n  Distinguished voluntary claims ('submitted', 190) from budget exhaustion ('exit_cost', 90).\n\n• High Dimensional Variance:\n  24 features across 12 repos with diverse build systems handled via LayerNorm scaling."
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_DARK

    add_card(s6, Inches(8.6), Inches(1.6), Inches(3.9), Inches(5.2), "The Surprising Dead End")
    tb = s6.shapes.add_textbox(Inches(8.8), Inches(2.2), Inches(3.5), Inches(4.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Dead End Explored:\n• Attempting to classify failure solely from git patch diff syntax.\n\nWhy It Failed:\n• An agent can submit syntactically pristine code that completely misses the bug.\n• Process dynamics (error loops, stalled turns) are essential to catch failure."
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_DARK

    # ==========================================================================
    # Slide 7: 3-Tier Results
    # ==========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Main Results: 3-Tier Benchmark Across 5 Seeds")

    # Embed PR/ROC curve image
    pr_path = "Model/figures/fig4_pr_roc_curves.png"
    if os.path.exists(pr_path):
        s7.shapes.add_picture(pr_path, Inches(0.8), Inches(1.6), Inches(6.8), Inches(3.2))

    add_card(s7, Inches(7.8), Inches(1.6), Inches(4.7), Inches(5.2), "Performance Analysis (§7)")
    tb = s7.shapes.add_textbox(Inches(8.0), Inches(2.2), Inches(4.3), Inches(4.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "1. Tier 2 vs. Tier 1 (Naive Heuristic):\n• PR-AUC: 0.976 vs. 0.964 (+0.012, p = 0.0007***)\n• F2 Score: 0.952 vs. 0.794 (+0.158)\n• Significant gain, bootstrap 95% CI excludes 0.\n\n2. Tier 2 vs. Tier 3 (Supervised XGBoost):\n• PR-AUC Gap: -0.0008 (p = 0.565, not significant)\n• Bootstrap 95% CI [-0.021, +0.030] includes 0.\n• Comparable to supervised ceiling without using failure labels!\n\n3. Latency & Training Overhead:\n• Latency: 0.004 ms / sample (<0.01 ms budget)\n• Training: 0.634 s total (lightweight & real-time)."
    p.font.size = Pt(10.5)
    p.font.color.rgb = COLOR_DARK

    # Add results summary table below image
    add_card(s7, Inches(0.8), Inches(5.0), Inches(6.8), Inches(1.8), "Benchmark Summary Table")
    tb = s7.shapes.add_textbox(Inches(1.0), Inches(5.4), Inches(6.4), Inches(1.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Tier 1 (Naive):     PR-AUC 0.964 ± 0.000 | F1 0.851 | F2 0.794 | 0.001 ms\nTier 2 (Ours):      PR-AUC 0.976 ± 0.003 | F1 0.947 | F2 0.952 | 0.004 ms\nTier 3 (Supervised): PR-AUC 0.977 ± 0.000 | F1 0.882 | F2 0.844 | 0.005 ms"
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY

    # ==========================================================================
    # Slide 8: Controlled Ablations
    # ==========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Controlled Ablations: Contamination & Feature Subsets")

    # Embed Ablation figure
    abl_path = "Model/figures/fig5_ablation_curves.png"
    if os.path.exists(abl_path):
        s8.shapes.add_picture(abl_path, Inches(0.8), Inches(1.6), Inches(6.8), Inches(3.2))

    add_card(s8, Inches(7.8), Inches(1.6), Inches(4.7), Inches(5.2), "Ablation Findings (§7.2)")
    tb = s8.shapes.add_textbox(Inches(8.0), Inches(2.2), Inches(4.3), Inches(4.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Ablation 1: Training Contamination (H2 Verified)\n• 0% Clean: PR-AUC = 0.9759 ± 0.0026\n• 1% Injected: PR-AUC = 0.9741 (-0.0017)\n• 10% Injected: PR-AUC = 0.9685 (-0.0074)\n• Drop exceeds seed std by 3x! Confirms Autoencoder learns to reconstruct anomalies when contaminated.\n\nAblation 2: Process vs. Patch Output (H1 Verified)\n• Full (D=24): PR-AUC 0.9759 | F2 0.9516\n• Process Dynamics (D=16): PR-AUC 0.9699 | F2 0.9532\n• Patch Only (D=8): PR-AUC 0.9636 (-0.012) | F2 0.9324\n• Process features capture critical failure dynamics!"
    p.font.size = Pt(10.5)
    p.font.color.rgb = COLOR_DARK

    add_card(s8, Inches(0.8), Inches(5.0), Inches(6.8), Inches(1.8), "Verdict on Hypotheses")
    tb = s8.shapes.add_textbox(Inches(1.0), Inches(5.4), Inches(6.4), Inches(1.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "• H1 (Process Dominance): SUPPORTED. Process dynamics significantly outperform patch diffs alone.\n• H2 (Contamination Degradation): SUPPORTED. Unlabeled failures compress anomaly reconstruction gap."
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY

    # ==========================================================================
    # Slide 9: Error Analysis & Interpretability
    # ==========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "In-Depth Error Analysis & Interpretability Attribution")

    add_card(s9, Inches(0.8), Inches(1.6), Inches(6.8), Inches(5.2), "Concrete Failure Cases (Threshold tau = 0.118)")
    tb = s9.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(6.4), Inches(4.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Case #1: False Positive (scikit-learn__scikit-learn-14894)\n• Score s(x) = 1.567 vs. tau = 0.118 (Distance +1.449)\n• Cause: Complex 40-step multi-file refactoring required wide directory searches, mimicking failure thrashing.\n\nCase #2: False Negative Near Miss (matplotlib__matplotlib-23562)\n• Score s(x) = 0.098 vs. tau = 0.118 (Distance -0.020)\n• Cause: Agent recovered from early error and submitted clean 1-line patch with subtle omission.\n\nCase #3: False Negative Inversion (sympy__sympy-15345)\n• Score s(x) = 0.045 vs. tau = 0.118 (Distance -0.073)\n• Cause: Premature 6-step exit with zero error signals.\n\nShared Blind Spot: Silent premature halts need prompt semantic verification."
    p.font.size = Pt(10)
    p.font.color.rgb = COLOR_DARK

    # Embed interpretability attribution image
    interp_path = "Model/figures/fig6_feature_attribution.png"
    if os.path.exists(interp_path):
        s9.shapes.add_picture(interp_path, Inches(7.8), Inches(1.6), Inches(4.7), Inches(5.2))

    # ==========================================================================
    # Slide 10: Practical Impact & Positioning
    # ==========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Practical Impact Translation & Industry Positioning")

    add_card(s10, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "Operational Savings (500 Tasks / Month)")
    tb = s10.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.2), Inches(4.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "In an enterprise CI/CD pipeline processing 500 tasks/mo:\n\n• Failure Catch Rate: Catches 380 / 400 failures (95.2% catch rate vs. Tier 1's 85.1%).\n\n• Silent Bug Escapes (FN): Reduced from 60 to 20 (-70% reduction in production escapes!).\n\n• False Audit Alerts (FP): Suppressed from 30 to 8 (-72% reduction in engineer triage toil!).\n\n• Net Monthly Savings: Delivering +$48,600/month net savings over naive heuristic baselines."
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_DARK

    add_card(s10, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "Positioning Against Monitoring Platforms")
    tb = s10.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "• Rule-Based Heuristics (LangSmith / ATFD Naive):\n  High false alarm rate (FPR > 20%), zero model adaptivity.\n\n• LLM Judges (Claude Sonnet 4 / GPT-5 in ATFD):\n  High semantic alignment, but incurs ~$0.05 per trace and 3-5 second evaluation latency.\n\n• Our Ghostwork Autoencoder:\n  Inference in 0.004 ms at ZERO dollar API cost, making it feasible for real-time CI/CD pre-commit hooks."
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_DARK

    # ==========================================================================
    # Slide 11: Conclusion & Contributions
    # ==========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Conclusion & Team Contributions")

    add_card(s11, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "Summary of Findings")
    tb = s11.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.2), Inches(4.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "• Research Question Answered:\n  Trajectory process dynamics reliably capture agent failure before code submission, outperforming static patch diffs by +0.012 PR-AUC and +0.019 F2.\n\n• Normal-Only Manifold:\n  Tier 2 Bottleneck Autoencoder achieves 0.976 PR-AUC, matching supervised XGBoost without requiring failure labels.\n\n• Future Work:\n  Integrate task-prompt semantic embeddings to detect premature zero-error exits (Case #3)."
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_DARK

    add_card(s11, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "Team Contributions & AI Statement")
    tb = s11.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Team Contributions (Team G07):\n• Dun-Jie Yang (Fergus4506): Data pipeline & EDA (§3-§4)\n• Charlie Wang (calculusfkyou): Autoencoder & Ablations (§5, §7)\n• Mo-Ty Wang (MO7YW4NG): 3-Tier benchmark & significance (§6-§8)\n• Wei-Chia Huang (lasvigas0701): Presentation deck & review (§1, §9)\n\nAI Usage Disclosure:\nClaude & Gemini assisted with matplotlib scripts and LaTeX formatting. All designs, experiments, and analyses were conducted and verified by the authors.\n\nFull Code & Reproduction:\nhttps://github.com/calculusfkyou/nycu-data-mining-assignment-2-115-1"
    p.font.size = Pt(10.5)
    p.font.color.rgb = COLOR_DARK

    # Save presentation
    output_path = OUTPUT_DIR / "HW2_G07_presentation.pptx"
    prs.save(output_path)
    print(f"Presentation saved to {output_path}")

    # Also save with root naming requirement
    prs.save("HW2_G07_presentation.pptx")
    print("Saved copy to HW2_G07_presentation.pptx")


if __name__ == "__main__":
    build_presentation()
