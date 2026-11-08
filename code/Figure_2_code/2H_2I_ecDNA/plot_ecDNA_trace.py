import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import mannwhitneyu, ttest_ind

#matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['svg.fonttype'] = 'none'

JOB_ID = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))
sample_list = ['3883', '5597']
JOB_SAMPLE = sample_list[JOB_ID]

targets = ['EGFR', 'PDGFRA']

palette_hex = [
    "#73b9a5", "#a33f3e", "#2261b2", "#fea17b", "#c5aedf",
    "#d4af9e", "#fea8d4", "#7e5679", "#6ea9fd", "#bd8985",
    "#d36d54", "#e4d177", "#89aeb3", "#fec4a7", "#977d89",
    "#637fd0", "#93581c", "#4bcbfe", "#82e2dd", "#dea257",
    "#70715d", "#009aa5", "#c27869", "#a893c6", "#00eafa",
    "#a96296", "#769be0", "#aa5839", "#7dcab8", "#847ebb",
    "#f28679", "#286958", "#00838d", "#b9989c", "#de8a5c",
    "#e3c7fe", "#97792d", "#6d6cb7", "#8e586f", "#b9d08a",
    "#a8a6e3", "#c35653", "#50729f", "#70548b", "#575a99",
    "#b26270", "#a58d39", "#9cc2c8", "#c0af85", "#72999a",
    "#5ebafe", "#59d6ac", "#984649", "#1bb1ca", "#c99b89",
    "#8fe7a1", "#568972", "#6690e7", "#e5b76b", "#798258",
    "#9984aa", "#a9dfc8", "#567319", "#cd8b68", "#979bb7",
    "#b17981", "#bacde7", "#8c6d80", "#576424", "#cfbe91",
    "#b2abca", "#e6bcb3", "#9783cf", "#2e9096", "#bd6c41",
]


N_BOOT = 1000
ALPHA = 0.05

NORMAL_TYPES = ['Microglia', 'Pericytes', 'Neuron GABA.', 'Neuron Glut.',
                'Olig. normal', 'Endothelial', 'T', 'AC']
MALIG_TYPES = ['AC.like', 'OPC.like Prolif.', 'OPC']

LEGEND_LABELS = {
    'normal': 'Normal',
    'malig': 'Tumor',
}

Y_MAX = 4.5
EGFR_VLINE_X = 245

PT_PER_INCH = 72.0

# Size presets in points: (width_pt, height_pt, label_fontsize, tick_fontsize,
# title_fontsize, chrom_fontsize, line_lw, vline_lw, chrom_vline_lw)
SIZE_PRESETS = {
    'square80pt': {
        'width_pt': 80, 'height_pt': 80,
        'label_fs': 7, 'tick_fs': 6, 'title_fs': 7,
        'chrom_fs': 4, 'line_lw': 0.6, 'vline_lw': 0.8, 'chrom_vline_lw': 0.4,
    },
    'wide342x40pt': {
        'width_pt': 342, 'height_pt': 40,
        'label_fs': 6, 'tick_fs': 5, 'title_fs': 6,
        'chrom_fs': 4, 'line_lw': 0.5, 'vline_lw': 0.7, 'chrom_vline_lw': 0.3,
    },
}


def bootstrap_ci(values, n_boot=1000, alpha=0.05):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    boot_means = [
        np.mean(np.random.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ]
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return np.mean(values), lower, upper


def precompute_group_traces(data, groups):
    """Compute mean + CI traces and order_val values once per group."""
    traces = {}
    group_values_at_order = {}
    order_val = 515

    for label, positions in groups.items():
        x_all, y_all = [], []
        for idx in positions:
            x_lists, y_lists = data[idx]
            x_all.extend(x_lists)
            y_all.extend(y_lists)

        if len(x_all) == 0:
            traces[label] = (np.array([]), np.array([]),
                             np.array([]), np.array([]))
            continue

        all_x = sorted(set(np.concatenate(x_all)))
        y_matrix = np.full((len(y_all), len(all_x)), np.nan)
        for k, (xs, ys) in enumerate(zip(x_all, y_all)):
            xs, ys = np.asarray(xs), np.asarray(ys)
            for j, x_val in enumerate(all_x):
                match = np.where(xs == x_val)[0]
                if len(match) > 0:
                    y_matrix[k, j] = ys[match[0]]

        mean_y, lower_y, upper_y = [], [], []
        for j in range(len(all_x)):
            m, lo, hi = bootstrap_ci(y_matrix[:, j], N_BOOT, ALPHA)
            mean_y.append(m)
            lower_y.append(lo)
            upper_y.append(hi)

        traces[label] = (
            np.array(all_x),
            np.array(mean_y),
            np.array(lower_y),
            np.array(upper_y),
        )

        if order_val in all_x:
            j_idx = all_x.index(order_val)
            col = y_matrix[:, j_idx]
            col = col[np.isfinite(col)]
            group_values_at_order[label] = col

    return traces, group_values_at_order, order_val


def make_plot(traces, change_points, target, sample, colors, preset, save_path):
    width_in = preset['width_pt'] / PT_PER_INCH
    height_in = preset['height_pt'] / PT_PER_INCH

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    ax.set_ylim(0, Y_MAX)

    for label, (xs, mean_y, lower_y, upper_y) in traces.items():
        if len(xs) == 0:
            continue
        ax.plot(xs, mean_y, color=colors[label], lw=preset['line_lw'],
                alpha=0.8, label=LEGEND_LABELS[label])
        ax.fill_between(xs, lower_y, upper_y, color=colors[label], alpha=0.15)

    ax.axvline(x=EGFR_VLINE_X, color='red', linestyle='--',
               lw=preset['vline_lw'], label=f'{target} locus')

    # --- chromosome change-point lines, with chromosome names as x tick labels ---
    cp_orders = change_points['order'].tolist()
    cp_chroms = change_points['bin_chromosome'].tolist()

    for cp_order in cp_orders:
        ax.axvline(cp_order, color='gray', linestyle='--',
                   alpha=0.5, lw=preset['chrom_vline_lw'])

    # place chromosome labels at the start of each chromosome
    ax.set_xticks(cp_orders)
    ax.set_xticklabels([str(c) for c in cp_chroms],
                       fontsize=preset['chrom_fs'])
    ax.tick_params(axis='x', length=0)  # no tick marks
    ax.tick_params(axis='y', labelsize=preset['tick_fs'])

    ax.set_xlabel('Genomic Position (Mb)', fontsize=preset['label_fs'])
    ax.set_ylabel(f'Min. Distance to {target} (um)',
                  fontsize=preset['label_fs'])
    ax.set_title(sample, fontsize=preset['title_fs'])

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


for target in targets:

    with open(f'{target}_distance_{JOB_SAMPLE}.pickle', 'rb') as f:
        data = pickle.load(f)
    with open(f'celltypes_for{target}_{JOB_SAMPLE}.pickle', 'rb') as f:
        celltypes = pickle.load(f)

    dna_loci = pd.read_csv(
        '/resnick/groups/CaiLab/jjfox/copied_from_central/gbm_dna_analysis/'
        '25-07-28_combined_dna/human_1k_genomic_loci_final.csv'
    )
    dna_loci['block_num'] = dna_loci['bin_name'].str.extract(r'block(\d+)_').astype(int)
    dna_loci['bin_num'] = dna_loci['bin_name'].str.extract(r'bin(\d+)').astype(int)
    dna_loci['order'] = ((dna_loci['block_num'] - 1) * 15) + dna_loci['bin_num']

    change_points = dna_loci.loc[
        dna_loci['bin_chromosome'] != dna_loci['bin_chromosome'].shift(),
        ['bin_chromosome', 'order']
    ].reset_index(drop=True)

    groups = {
        'normal': np.where(np.isin(celltypes, NORMAL_TYPES))[0],
        'malig':  np.where(np.isin(celltypes, MALIG_TYPES))[0],
    }
    colors = {
        'normal': palette_hex[0],
        'malig':  palette_hex[1],
    }

    traces, group_values_at_order, order_val = precompute_group_traces(data, groups)

    if 'normal' in group_values_at_order and 'malig' in group_values_at_order:
        n_vals = group_values_at_order['normal']
        m_vals = group_values_at_order['malig']

        mw_stat, mw_pval = mannwhitneyu(n_vals, m_vals, alternative='two-sided')
        t_stat, t_pval = ttest_ind(n_vals, m_vals, equal_var=False)

        print(f"Sample={JOB_SAMPLE}, Order={order_val}, "
              f"n_normal={len(n_vals)}, n_malig={len(m_vals)}")
        print(f"  Mann-Whitney: U={mw_stat:.1f}, p={mw_pval:.4e}")
        print(f"  Welch t-test: t={t_stat:.4f}, p={t_pval:.4e}")

    save_dir = '../../../figures/Figure_S2/Figure_S2F/'
    os.makedirs(save_dir, exist_ok=True)

    for size_name, preset in SIZE_PRESETS.items():
        save_path = (f'{save_dir}'
                     f'PDGFRA_{JOB_SAMPLE}_normalvsmalignant_{target}'
                     f'_proximity_trace_{size_name}.svg')
        make_plot(traces, change_points, target, JOB_SAMPLE,
                  colors, preset, save_path)

    # --- Export plot legend as separate figure ---
    fig_leg, ax_leg = plt.subplots(figsize=(3, 2))
    ax_leg.plot([], [], color=colors['normal'], lw=2, label='Normal')
    ax_leg.plot([], [], color=colors['malig'], lw=2, label='Tumor')
    ax_leg.axvline(x=0, color='red', linestyle='--', lw=1.5, label=f'{target} locus')
    ax_leg.legend(loc='center', frameon=False, fontsize=14)
    ax_leg.axis('off')
    fig_leg.tight_layout()
    fig_leg.savefig(f'{save_dir}proximity_trace_legend.svg', bbox_inches='tight')
    plt.close(fig_leg)