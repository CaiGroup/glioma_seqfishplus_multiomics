import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import TwoSlopeNorm
import matplotlib.cm as cm
from scipy.stats import mannwhitneyu, ttest_ind

#matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 14

JOB_ID = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))
sample_list = ['4555','3557','2543', '2683', '5121', '5597', '3883']
JOB_SAMPLE = sample_list[JOB_ID]

targets = ['CDKN2A']

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
    'normal': 'Normal Cells',
    'malig': 'Tumor Cells',
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

    # get CNV profile for chromosome
    df = pd.read_csv('/home/jjfox/resnick_ci2_gbm_/Data/DNA_data/CNV_mat.csv')
    df.columns = df.columns.str.replace('.', '_', regex=False)
    bin_to_order = dict(zip(dna_loci['bin_name'],dna_loci['order']))
    df.rename(columns=bin_to_order, inplace=True)
    sample_cnv = df.loc[df['ID'] == JOB_SAMPLE]
    cnvs = sample_cnv.iloc[0].loc[509:555] # change orders to match chromosome
    
    groups = {
        'normal': np.where(np.isin(celltypes, NORMAL_TYPES))[0],
        'malig':  np.where(np.isin(celltypes, MALIG_TYPES))[0],
    }
    colors = {
        'normal': palette_hex[0],
        'malig':  palette_hex[1],
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    order_val = 515
    group_values_at_order = {}

    for label, positions in groups.items():
        x_all, y_all = [], []
        for idx in positions:
            x_lists, y_lists = data[idx]
            x_all.extend(x_lists)
            y_all.extend(y_lists)

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

        mean_y = np.array(mean_y)
        lower_y = np.array(lower_y)
        upper_y = np.array(upper_y)

        ax.plot(all_x, mean_y, color=colors[label], lw=1, alpha=0.8,
                label=f'{LEGEND_LABELS[label]} (n={len(y_all)})')
        ax.fill_between(all_x, lower_y, upper_y, color=colors[label], alpha=0.15)

        if order_val in all_x:
            j_idx = all_x.index(order_val)
            col = y_matrix[:, j_idx]
            col = col[np.isfinite(col)]
            group_values_at_order[label] = col

    ax.axvline(x=515, color='red', linestyle='--', lw=1.5, label='CDKN2A locus')

    if 'normal' in group_values_at_order and 'malig' in group_values_at_order:
        n_vals = group_values_at_order['normal']
        m_vals = group_values_at_order['malig']

        mw_stat, mw_pval = mannwhitneyu(n_vals, m_vals, alternative='two-sided')
        t_stat, t_pval = ttest_ind(n_vals, m_vals, equal_var=False)

        print(f"Sample={JOB_SAMPLE}, Order={order_val}, "
              f"n_normal={len(n_vals)}, n_malig={len(m_vals)}")
        print(f"  Mann-Whitney: U={mw_stat:.1f}, p={mw_pval:.4e}")
        print(f"  Welch t-test: t={t_stat:.4f}, p={t_pval:.4e}")

    # --- CNV colorbar at top ---
    cmap_cnv = cm.RdBu_r
    norm_cnv = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)

    cbar_ax = ax.inset_axes([0, 1.02, 1, 0.05])
    x_vals = cnvs.index.values.astype(float)
    colors_mapped = cmap_cnv(norm_cnv(cnvs.values.astype(float)))

    for i in range(len(x_vals) - 1):
        cbar_ax.axvspan(x_vals[i], x_vals[i + 1], color=colors_mapped[i])

    cbar_ax.set_xlim(509, 554)
    cbar_ax.set_yticks([])
    cbar_ax.set_xticks([])
    cbar_ax.spines[:].set_visible(False)

    # --- CNV colormap legend on right ---
    sm = cm.ScalarMappable(cmap=cmap_cnv, norm=norm_cnv)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, location='right', shrink=0.3, pad=0.02, aspect=10)
    cbar.set_ticks([-2, -1, 0, 1, 2])
    cbar.set_label('CNV', fontsize=14)

    ax.set_xlabel('Chr9 (Mb)', fontsize=28)
    ax.set_ylabel(f'distance to nearest\n{target} dot (um)', fontsize=28)
    ax.set_title(JOB_SAMPLE, fontsize=28)
    ax.tick_params(axis='both', labelsize=28)
    ax.set_ylim(0, 7)
    ax.set_xlim(509, 554)
    ax.legend(loc='lower right', frameon=False)
    fig.tight_layout()

    save_path = (f'../../../figures/Figure_2/Figure_2C/'
                 f'cnv_chr9_{JOB_SAMPLE}_normalvsmalignant_{target}_proximity_trace.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)