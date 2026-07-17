#!/usr/bin/env python3
"""
Generate publication-quality charts comparing RAW, MINCO-COLD, and MINCO-HOT
navigation methods for the NavDP research paper.

Data source: Experimental results from Table 1-7.
All charts saved to /home/alioth/NavDP/results/navdp_minco_static_real/analysis_charts/
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── Global style configuration ──────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.color': '#cccccc',
    'grid.linestyle': '-',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
})

OUTPUT_DIR = '/home/alioth/NavDP/results/navdp_minco_static_real/analysis_charts'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Color scheme ────────────────────────────────────────────────────────────────
COLORS = {
    'RAW':        '#4472C4',
    'MINCO-COLD': '#ED7D31',
    'MINCO-HOT':  '#70AD47',
}
COLOR_LIST = [COLORS['RAW'], COLORS['MINCO-COLD'], COLORS['MINCO-HOT']]
METHODS = ['RAW', 'MINCO-COLD', 'MINCO-HOT']

# ── Helper functions ────────────────────────────────────────────────────────────

def add_value_labels(bars, fmt='{:.1f}', offset=0.02, fontweight='normal'):
    """Add value labels above bar chart bars."""
    for bar in bars:
        h = bar.get_height()
        if np.isfinite(h):
            ax = bar.axes
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + offset * ax.get_ylim()[1],
                fmt.format(h),
                ha='center', va='bottom', fontsize=9, fontweight=fontweight
            )

def ci_string(ci_low, ci_high, fmt='{:.3f}'):
    """Format a confidence interval string."""
    return f'[{fmt.format(ci_low)}, {fmt.format(ci_high)}]'

# ── 1. Success Rate Comparison ──────────────────────────────────────────────────
def fig1_success_rate():
    fig, ax = plt.subplots(figsize=(7, 5))
    vals = [60.0, 45.0, 65.0]
    ci_low  = [38.7, 25.8, 43.3]
    ci_high = [78.1, 65.8, 81.9]
    yerr_low  = np.array(vals) - np.array(ci_low)
    yerr_high = np.array(ci_high) - np.array(vals)

    x = np.arange(len(METHODS))
    bars = ax.bar(x, vals, width=0.55, color=COLOR_LIST, edgecolor='white', linewidth=0.5)
    ax.errorbar(x, vals, yerr=[yerr_low, yerr_high],
                fmt='none', capsize=5, capthick=1.2, ecolor='#333333', elinewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(METHODS)
    ax.set_ylabel('Success Rate (%)')
    ax.set_ylim(0, 100)
    ax.set_title('Success Rate by Navigation Method')
    add_value_labels(bars, fmt='{:.0f}', offset=2.5)
    fig.subplots_adjust(top=0.88, bottom=0.12)
    fig.savefig(os.path.join(OUTPUT_DIR, 'success_rate_comparison.png'))
    plt.close(fig)
    print('[1/10] success_rate_comparison.png')

# ── 2. Tracking Error Comparison ────────────────────────────────────────────────
def fig2_tracking_error():
    fig, ax = plt.subplots(figsize=(7, 5))
    vals = [0.0446, 0.0356, 0.0302]
    ci_low  = [0.0398, 0.0237, 0.0217]
    ci_high = [0.0518, 0.0479, 0.0401]
    yerr_low  = np.array(vals) - np.array(ci_low)
    yerr_high = np.array(ci_high) - np.array(vals)

    x = np.arange(len(METHODS))
    bars = ax.bar(x, vals, width=0.55, color=COLOR_LIST, edgecolor='white', linewidth=0.5)
    ax.errorbar(x, vals, yerr=[yerr_low, yerr_high],
                fmt='none', capsize=5, capthick=1.2, ecolor='#333333', elinewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(METHODS)
    ax.set_ylabel('Cross-Track Error RMSE (m)')
    ax.set_title('Tracking Error by Navigation Method')
    ax.set_ylim(0, 0.07)
    add_value_labels(bars, fmt='{:.4f}', offset=0.002)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'tracking_error_comparison.png'))
    plt.close(fig)
    print('[2/10] tracking_error_comparison.png')

# ── 3. Navigation Metrics Radar (normalized to RAW baseline) ────────────────────
def fig3_navigation_radar():
    """Show 4 navigation metrics normalized to RAW=100%, with improvement direction."""
    methods = ['RAW', 'MINCO-COLD', 'MINCO-HOT']
    # Normalise: RAW = 100% for all, and show relative change.
    # Metrics: Success Rate (higher better), SPL (higher better),
    #          Tracking RMSE (lower better), Episode Duration (lower better)
    # Values as percentages of RAW baseline.
    success = [100.0, 45.0/60.0*100, 65.0/60.0*100]   # 100, 75, 108.3
    spl     = [100.0, 0.437/0.561*100, 0.612/0.561*100] # 100, 77.9, 109.1
    rmse    = [100.0, 0.0356/0.0446*100, 0.0302/0.0446*100] # 100, 79.8, 67.7
    dur     = [100.0, 44.02/40.23*100, 44.01/40.23*100] # 100, 109.4, 109.4

    metric_names = ['Success Rate\n(higher better)', 'SPL\n(higher better)',
                    'Tracking RMSE\n(lower better)',  'Episode Duration\n(lower better)']

    x = np.arange(len(metric_names))
    width = 0.22
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (m, c) in enumerate(zip(methods, COLOR_LIST)):
        if m == 'RAW':
            vals = [success[0], spl[0], rmse[0], dur[0]]
        elif m == 'MINCO-COLD':
            vals = [success[1], spl[1], rmse[1], dur[1]]
        else:
            vals = [success[2], spl[2], rmse[2], dur[2]]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=m, color=c, edgecolor='white', linewidth=0.5)

    ax.axhline(y=100, color='#999999', linestyle='--', linewidth=0.8, label='RAW baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel('Relative to RAW Baseline (%)')
    ax.set_title('Navigation Metrics Relative to RAW Baseline')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(0, 130)

    # Add "better" / "worse" directional markers
    for i in range(len(metric_names)):
        pass  # direction is indicated in x-tick labels

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'navigation_metrics_radar.png'))
    plt.close(fig)
    print('[3/10] navigation_metrics_radar.png')

# ── 4. Smoothness Comparison ────────────────────────────────────────────────────
def fig4_smoothness():
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 5))

    # Left: Equivalent Jerk RMS
    jerk_vals = [0.650, 0.371]
    jerk_ci_low  = [0.609, 0.347]
    jerk_ci_high = [0.694, 0.395]
    jerk_yerr_low  = np.array(jerk_vals) - np.array(jerk_ci_low)
    jerk_yerr_high = np.array(jerk_ci_high) - np.array(jerk_vals)
    j_methods = ['MINCO-COLD', 'MINCO-HOT']
    j_colors  = [COLORS['MINCO-COLD'], COLORS['MINCO-HOT']]
    x = np.arange(len(j_methods))
    bars = ax_l.bar(x, jerk_vals, width=0.45, color=j_colors, edgecolor='white', linewidth=0.5)
    ax_l.errorbar(x, jerk_vals, yerr=[jerk_yerr_low, jerk_yerr_high],
                  fmt='none', capsize=5, capthick=1.2, ecolor='#333333', elinewidth=1.2)
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(j_methods)
    ax_l.set_ylabel('Equivalent Jerk RMS (m/s$^3$)')
    ax_l.set_title('(a) Jerk Smoothness')
    add_value_labels(bars, fmt='{:.3f}', offset=0.015)
    ax_l.set_ylim(0, 0.85)

    # Right: Yaw Rate RMS
    yaw_vals = [0.103, 0.122]
    yaw_ci_low  = [0.099, 0.117]
    yaw_ci_high = [0.108, 0.127]
    yaw_yerr_low  = np.array(yaw_vals) - np.array(yaw_ci_low)
    yaw_yerr_high = np.array(yaw_ci_high) - np.array(yaw_vals)
    bars = ax_r.bar(x, yaw_vals, width=0.45, color=j_colors, edgecolor='white', linewidth=0.5)
    ax_r.errorbar(x, yaw_vals, yerr=[yaw_yerr_low, yaw_yerr_high],
                  fmt='none', capsize=5, capthick=1.2, ecolor='#333333', elinewidth=1.2)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(j_methods)
    ax_r.set_ylabel('Yaw Rate RMS (rad/s)')
    ax_r.set_title('(b) Yaw Rate Smoothness')
    add_value_labels(bars, fmt='{:.3f}', offset=0.003)
    ax_r.set_ylim(0, 0.16)

    fig.suptitle('Smoothness Metrics Comparison', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'smoothness_comparison.png'))
    plt.close(fig)
    print('[4/10] smoothness_comparison.png')

# ── 5. Planning Reliability ─────────────────────────────────────────────────────
def fig5_planning_reliability():
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Plan Publish Rate (all 3), Optimizer Success Rate (MINCO only),
    # Validation Success Rate (MINCO only)
    categories = ['Plan Publish Rate', 'Optimizer Success Rate', 'Validation Success Rate']
    # RAW has N/A for optimizer and validation; show as 0 for stacking but label NA
    raw_vals    = [99.27, 0, 0]
    cold_vals   = [81.14, 96.77, 98.92]
    hot_vals    = [84.20, 96.04, 97.74]

    x = np.arange(len(categories))
    width = 0.22

    bars1 = ax.bar(x - width, raw_vals,  width, label='RAW',       color=COLORS['RAW'],       edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x,        cold_vals, width, label='MINCO-COLD', color=COLORS['MINCO-COLD'], edgecolor='white', linewidth=0.5)
    bars3 = ax.bar(x + width, hot_vals,  width, label='MINCO-HOT',  color=COLORS['MINCO-HOT'],  edgecolor='white', linewidth=0.5)

    # Mark N/A on RAW for optimizer and validation
    for i in [1, 2]:
        ax.text(x[i] - width, 3, 'N/A', ha='center', va='bottom', fontsize=8,
                color='#666666', fontstyle='italic')

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Rate (%)')
    ax.set_ylim(0, 110)
    ax.set_title('Planning Pipeline Reliability')
    ax.legend(loc='lower left', framealpha=0.9)

    # Value labels on non-NA bars
    for bars, vals in [(bars1, raw_vals), (bars2, cold_vals), (bars3, hot_vals)]:
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                        f'{v:.1f}%' if v < 100 else f'{v:.2f}%',
                        ha='center', va='bottom', fontsize=7.5)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'planning_reliability.png'))
    plt.close(fig)
    print('[5/10] planning_reliability.png')

# ── 6. Safety Analysis ──────────────────────────────────────────────────────────
def fig6_safety():
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9, 4.5))

    # Left: Raw Min Clearance
    min_clr = 0.515  # mid-point of [0.496, 0.534]
    min_clr_lo = 0.496
    min_clr_hi = 0.534
    ax_l.bar(['Raw Min Clearance'], [min_clr], width=0.45, color='#5B9BD5',
             edgecolor='white', linewidth=0.5)
    ax_l.errorbar(['Raw Min Clearance'], [min_clr],
                  yerr=[[min_clr - min_clr_lo], [min_clr_hi - min_clr]],
                  fmt='none', capsize=5, capthick=1.2, ecolor='#333333', elinewidth=1.2)
    ax_l.set_ylabel('Minimum Clearance (m)')
    ax_l.set_ylim(0, 0.7)
    ax_l.set_title('(a) Raw Plan Minimum Clearance')
    ax_l.text(0, min_clr + 0.02, f'{min_clr:.3f} m', ha='center', va='bottom', fontsize=10)

    # Right: Raw Unsafe Ratio
    unsafe_mid = 4.575  # mid of [3.78, 5.37]
    unsafe_lo = 3.78
    unsafe_hi = 5.37
    ax_r.bar(['Raw Unsafe Ratio'], [unsafe_mid], width=0.45, color='#C55A5A',
             edgecolor='white', linewidth=0.5)
    ax_r.errorbar(['Raw Unsafe Ratio'], [unsafe_mid],
                  yerr=[[unsafe_mid - unsafe_lo], [unsafe_hi - unsafe_mid]],
                  fmt='none', capsize=5, capthick=1.2, ecolor='#333333', elinewidth=1.2)
    ax_r.set_ylabel('Unsafe Ratio (%)')
    ax_r.set_ylim(0, 8)
    ax_r.set_title('(b) Raw Plan Unsafe Ratio')
    ax_r.text(0, unsafe_mid + 0.2, f'{unsafe_mid:.2f}%', ha='center', va='bottom', fontsize=10)

    fig.suptitle('Safety Characteristics of Raw NavDP Plans', y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'safety_analysis.png'))
    plt.close(fig)
    print('[6/10] safety_analysis.png')

# ── 7. Method Comparison Summary (2x2) ──────────────────────────────────────────
def fig7_summary():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    data = {
        'Success Rate (%)': {
            'vals': [60.0, 45.0, 65.0],
            'ci_low':  [38.7, 25.8, 43.3],
            'ci_high': [78.1, 65.8, 81.9],
            'ylim': (0, 100),
            'ylabel': 'Success Rate (%)',
            'fmt': '{:.0f}',
            'offset': 3,
        },
        'SPL': {
            'vals': [0.561, 0.437, 0.612],
            'ci_low':  [0.367, 0.241, 0.420],
            'ci_high': [0.750, 0.635, 0.801],
            'ylim': (0, 1.0),
            'ylabel': 'SPL',
            'fmt': '{:.3f}',
            'offset': 0.02,
        },
        'Tracking RMSE (m)': {
            'vals': [0.0446, 0.0356, 0.0302],
            'ci_low':  [0.0398, 0.0237, 0.0217],
            'ci_high': [0.0518, 0.0479, 0.0401],
            'ylim': (0, 0.07),
            'ylabel': 'Cross-Track Error RMSE (m)',
            'fmt': '{:.4f}',
            'offset': 0.002,
        },
        'Episode Duration (s)': {
            'vals': [40.23, 44.02, 44.01],
            'ci_low':  [33.37, 26.48, 32.96],
            'ci_high': [46.99, 67.88, 55.25],
            'ylim': (0, 80),
            'ylabel': 'Duration (s)',
            'fmt': '{:.1f}',
            'offset': 2,
        },
    }

    titles = ['(a) Success Rate', '(b) SPL', '(c) Tracking RMSE', '(d) Episode Duration']

    for idx, ((key, d), title) in enumerate(zip(data.items(), titles)):
        ax = axes[idx // 2][idx % 2]
        vals = d['vals']
        yerr_low  = np.array(vals) - np.array(d['ci_low'])
        yerr_high = np.array(d['ci_high']) - np.array(vals)

        x = np.arange(len(METHODS))
        bars = ax.bar(x, vals, width=0.55, color=COLOR_LIST, edgecolor='white', linewidth=0.5)
        ax.errorbar(x, vals, yerr=[yerr_low, yerr_high],
                    fmt='none', capsize=4, capthick=1.0, ecolor='#333333', elinewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS, fontsize=9)
        ax.set_ylabel(d['ylabel'])
        ax.set_ylim(d['ylim'])
        ax.set_title(title)
        add_value_labels(bars, fmt=d['fmt'], offset=d['offset'])

    fig.subplots_adjust(top=0.93, bottom=0.08, left=0.08, right=0.96, hspace=0.35, wspace=0.30)
    fig.savefig(os.path.join(OUTPUT_DIR, 'method_comparison_summary.png'))
    plt.close(fig)
    print('[7/10] method_comparison_summary.png')

# ── 8. Tracking RMSE Reduction (Waterfall) ──────────────────────────────────────
def fig8_tracking_rmse_reduction():
    fig, ax = plt.subplots(figsize=(8, 5))

    # Waterfall: start at RAW, then step down
    raw_val = 0.0446
    cold_val = 0.0356
    hot_val = 0.0302

    # Percentage reductions
    cold_reduction_pct = 20.2
    hot_reduction_pct = 32.2

    categories = ['RAW\n(baseline)', 'MINCO-COLD\n(-20.2%)', 'MINCO-HOT\n(-32.2%)']
    vals = [raw_val, cold_val, hot_val]
    colors_waterfall = [COLORS['RAW'], COLORS['MINCO-COLD'], COLORS['MINCO-HOT']]

    x = np.arange(len(categories))
    bars = ax.bar(x, vals, width=0.5, color=colors_waterfall, edgecolor='white', linewidth=0.5)

    # Add connecting lines to show reduction
    for i in range(len(categories) - 1):
        ax.plot([i + 0.25, i + 0.75], [vals[i], vals[i+1]], color='#555555',
                linewidth=1.5, linestyle='--', zorder=0)

    # Add delta annotations
    deltas = [0, -0.0090, -0.0054]  # 0.0446-0.0356=0.0090, 0.0356-0.0302=0.0054
    for i in range(1, len(categories)):
        mid_y = (vals[i-1] + vals[i]) / 2
        ax.annotate(
            f'Δ = {deltas[i]:.4f} m',
            xy=(i - 0.3, mid_y),
            fontsize=8, color='#666666',
            ha='center', va='bottom',
            rotation=0,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Cross-Track Error RMSE (m)')
    ax.set_title('Tracking RMSE Reduction via Post-Processing')
    ax.set_ylim(0, 0.06)

    # Value labels on bars
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0015,
                f'{v:.4f}', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'tracking_rmse_reduction.png'))
    plt.close(fig)
    print('[8/10] tracking_rmse_reduction.png')

# ── 9. Warm Start Analysis (donut) ──────────────────────────────────────────────
def fig9_warm_start():
    fig, ax = plt.subplots(figsize=(6, 5))

    accept = 98.50
    reject = 100.0 - accept  # 1.5%

    sizes = [accept, reject]
    colors_donut = [COLORS['MINCO-HOT'], '#D6D6D6']
    labels = [f'Accepted: {accept:.1f}%', f'Rejected: {reject:.1f}%']
    explode = (0.03, 0)

    wedges, texts = ax.pie(
        sizes, explode=explode, labels=labels,
        colors=colors_donut, startangle=90,
        textprops={'fontsize': 10, 'fontfamily': 'DejaVu Serif'},
        autopct=None,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'width': 0.4},
    )

    # Inner text
    ax.text(0, 0, f'{accept:.1f}%', ha='center', va='center',
            fontsize=18, fontweight='bold', color=COLORS['MINCO-HOT'],
            fontfamily='DejaVu Serif')

    ax.set_title('MINCO-HOT Warm Start Acceptance Rate\n(Hot Accept Rate: 98.5%)',
                 fontsize=12, pad=20)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'warm_start_analysis.png'))
    plt.close(fig)
    print('[9/10] warm_start_analysis.png')

# ── 10. Executive Summary Table ─────────────────────────────────────────────────
def fig10_executive_summary():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    # Build table data
    col_labels = ['Metric', 'RAW', 'MINCO-COLD', 'MINCO-HOT']

    rows = [
        ['Success Rate (%)',     '60.0 [38.7, 78.1]', '45.0 [25.8, 65.8]', '65.0 [43.3, 81.9]'],
        ['Collision Rate (%)',   '0.0',                '0.0',                '0.0'],
        ['Tracking RMSE (m)',    '0.0446 [0.0398,\n0.0518]', '0.0356 [0.0237,\n0.0479]', '0.0302 [0.0217,\n0.0401]'],
        ['Episode Duration (s)', '40.23 [33.37,\n46.99]',  '44.02 [26.48,\n67.88]', '44.01 [32.96,\n55.25]'],
        ['Path Length (m)',      '13.75 [11.00,\n16.34]', '10.39 [6.89,\n14.21]', '14.89 [10.87,\n19.02]'],
        ['SPL',                  '0.561 [0.367,\n0.750]', '0.437 [0.241,\n0.635]', '0.612 [0.420,\n0.801]'],
        ['Plan Publish Rate (%)','99.27',               '81.14',              '84.20'],
        ['Planning Mean (ms)',   '242.58',              '26.26',              '25.94'],
        ['Planning P95 (ms)',    '272.03',              '287.96',             '285.38'],
    ]

    # Determine best values per metric (for bolding)
    # Higher is better: Success Rate, SPL, Plan Publish Rate
    # Lower is better: Tracking RMSE, Episode Duration, Path Length, Planning Mean, Planning P95
    best_cols = {
        0: [2, 2, None, None, None, 2, 0, 2, 0],  # column index of best value for each row
        # manual override: row 0 (Success Rate high=MINCO-HOT col2),
        # row 1 (Collision Rate all equal),
        # row 2 (Tracking RMSE low=MINCO-HOT col2),
        # row 3 (Duration low=RAW col0),
        # row 4 (Path Length low=MINCO-COLD col1),
        # row 5 (SPL high=MINCO-HOT col2),
        # row 6 (Plan Publish Rate high=RAW col0),
        # row 7 (Planning Mean low=MINCO-HOT col2),
        # row 8 (Planning P95 low=MINCO-COLD col1) -- wait: 272.03 vs 287.96 vs 285.38 -> RAW is lowest
    }
    # Let me re-check:
    # Success Rate: highest=MINCO-HOT=65.0 (col 2 -> index 3 in table)
    # Collision: all 0.0 (none)
    # RMSE: lowest=MINCO-HOT=0.0302 (col 2)
    # Duration: lowest=RAW=40.23 (col 0)
    # Path Length: lowest=MINCO-COLD=10.39 (col 1)
    # SPL: highest=MINCO-HOT=0.612 (col 2)
    # Plan Publish Rate: highest=RAW=99.27 (col 0)
    # Planning Mean: lowest=MINCO-HOT=25.94 (col 2)
    # Planning P95: lowest=RAW=272.03 (col 0)
    best_indices = {
        0: 2, 1: None, 2: 2, 3: 0, 4: 1, 5: 2, 6: 0, 7: 2, 8: 0
    }

    # Build cell text with coloring
    cell_texts = [col_labels]
    for i, row in enumerate(rows):
        cell_row = [row[0]]
        for j in range(1, 4):
            cell_row.append(row[j])
        cell_texts.append(cell_row)

    # Prepare cell colors (white background, light highlight on best)
    row_colors = []
    header_color = ['#2F5496'] * 4  # dark blue header
    for i, row in enumerate(rows):
        best_col = best_indices.get(i)
        rcols = ['white'] * 4
        if best_col is not None:
            rcols[best_col + 1] = '#E2EFDA'  # light green for best
        row_colors.append(rcols)

    all_colors = [header_color] + row_colors

    # Create the table
    tbl = ax.table(
        cellText=cell_texts,
        cellLoc='center',
        loc='center',
        colWidths=[0.20, 0.22, 0.22, 0.22],
    )

    # Style
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2F5496')
            cell.set_text_props(color='white', fontweight='bold', fontsize=10)
        else:
            cell.set_facecolor(all_colors[row][col] if col < len(all_colors[row]) else 'white')
            cell.set_text_props(fontsize=8.5)
        cell.set_edgecolor('#CCCCCC')
        cell.set_linewidth(0.5)
        cell.PAD = 0.05

    ax.set_title('Executive Summary: Navigation Performance Metrics',
                 fontsize=13, fontweight='bold', pad=15)

    # Add footnote
    fig.text(0.5, 0.01, 'Values shown as mean [95% Wilson confidence interval]. '
             'Green highlight indicates best value per metric.',
             ha='center', fontsize=8, fontstyle='italic', color='#555555')

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'executive_summary_table.png'))
    plt.close(fig)
    print('[10/10] executive_summary_table.png')


# ── Main ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    np.random.seed(42)

    fig1_success_rate()
    fig2_tracking_error()
    fig3_navigation_radar()
    fig4_smoothness()
    fig5_planning_reliability()
    fig6_safety()
    fig7_summary()
    fig8_tracking_rmse_reduction()
    fig9_warm_start()
    fig10_executive_summary()

    print('\nAll 10 charts generated in:')
    print(f'  {OUTPUT_DIR}')
