import glob
import numpy as np
import pandas as pd
#import tqdm
import os
import re
import pickle
import sys
sys.setrecursionlimit(50000)
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy import stats
from collections import defaultdict

JOB_ID = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))
sample_list = ['4555','3557', '2543', '2683', '5121', '5597', '3883']
JOB_SAMPLE = sample_list[JOB_ID]

# define target
targets = ['CDKN2A']

# choose plots
plot_normal_v_malignant = False

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

for target in targets:
    # load data
    data_path = '/resnick/groups/CaiLab/jjfox/copied_from_central/gbm_dna_analysis/25-07-28_combined_dna/DNA_data_table.csv'
    combined_df = pd.read_csv(data_path, low_memory=False)
    
    # Rename column for cell ID
    combined_df.rename(columns={'Unnamed: 0': 'cell_id'}, inplace=True)
    # make diagnosis column
    combined_df['diagnosis'] = combined_df['type_2'] + '_' + combined_df['type_1']
    # replace . with _ in column names
    combined_df.columns = [
        col.replace('.', '_') for col in combined_df.columns
    ]
    # rename 45S rDNA Column
    combined_df.rename(columns={'X45S_rDNA': '45S_rDNA'}, inplace=True)
    
    # only keep blockbin columns
    
    cluster_df = combined_df.set_index('cell_id')
    
    # keep blockbins
    cluster_df_bb = cluster_df.iloc[:, :-29]
    # keep serial
    cluster_df_ser = cluster_df.iloc[: , -23:-14]
    
    # only keep cells with more than 200 unique bins
    
    # Keep rows where number of nonzero values >= defined number
    cluster_df_bb = cluster_df_bb[cluster_df_bb.astype(bool).sum(axis=1) >= 250]
    
    # combine
    cluster_df_selected_columns = pd.concat([cluster_df_bb, cluster_df_ser], join = 'inner',  axis = 1)
    
    # convert to chromosome arms
    
    # transpose blockbin df for blockbins as rows
    cluster_df_bb_T = cluster_df_bb.T
    
    # load dictionary mapping chromosome arms to blockbins
    with open('/resnick/groups/CaiLab/jjfox/copied_from_central/gbm_dna_analysis/25-07-28_combined_dna/chrom_arm_dict.pkl', 'rb') as f:
        chrom_arm_dict = pickle.load(f)
    
    # map chromosome arms for blockbins
    cluster_df_bb_T['chrom_arm'] = cluster_df_bb_T.index.map(chrom_arm_dict)
    
    # map chromosome number for blockbins
    
    # make dictionary
    probe_df = pd.read_csv("/resnick/groups/CaiLab/jjfox/copied_from_central/DNA_seqFISH/human/24-12-19_human_genome_DNA_seqFISH_final_probes.csv")
    df_unique = probe_df.drop_duplicates(subset='bin_name', keep='last')
    chr_dict = dict(zip(df_unique['bin_name'], df_unique['chr']))
    
    # map
    cluster_df_bb_T['chromosome'] = cluster_df_bb_T.index.map(chr_dict)
    
    # make single column of chr and arm
    cluster_df_bb_T['chrom_arm_combined'] = cluster_df_bb_T['chromosome'] + '_' + cluster_df_bb_T['chrom_arm']
    
    # drop unnecessary columns
    df_filtered = cluster_df_bb_T.drop(columns=['chrom_arm', 'chromosome'])
    
    # Group by 'chrom_arm_combined' and sum the remaining columns
    grouped_counts = df_filtered.groupby('chrom_arm_combined').sum(numeric_only=True)
    
    # transpose for rows as cells
    grouped_counts_T = grouped_counts.T
    
    # combine with serial genes
    grouped_df_selected_columns = pd.concat([grouped_counts_T, cluster_df_ser], join = 'inner', axis = 1)
    
    # look at specific sample
    path = '/resnick/groups/mthomson/ci2_data/CaiLab/gbm/Data/DNA_data/'
    
    # get all files
    files = glob.glob(f"{path}/*assigned.csv")
    
    # only keep JOB_SAMPLE
    patient_id = JOB_SAMPLE
    if '_' in patient_id:
        patient_id = patient_id.split('_')[0]
    files = [i for i in files if patient_id in i]
    
    # load data
    df_1 = pd.read_csv(files[0], low_memory = False)
    
    # fix chromosomes
    # make dictionary
    probe_df = pd.read_csv("/resnick/groups/CaiLab/jjfox/copied_from_central/DNA_seqFISH/human/24-12-19_human_genome_DNA_seqFISH_final_probes.csv")
    df_unique = probe_df.drop_duplicates(subset='bin_name', keep='last')
    chr_dict = dict(zip(df_unique['bin_name'], df_unique['chr']))
    
    # map
    df_1["blockbin"] = (
        df_1["block_assigned"].str.replace("block_", "block", regex=False)
        + "_" +
        df_1["name"].str.replace("bin_", "bin", regex=False)
    )
    df_1['chromosome_true'] = df_1.blockbin.map(chr_dict)
    
    # map chromosome arms
    # load dictionary mapping chromosome arms to blockbins
    with open('/resnick/groups/CaiLab/jjfox/copied_from_central/gbm_dna_analysis/25-07-28_combined_dna/chrom_arm_dict.pkl', 'rb') as f:
        chrom_arm_dict = pickle.load(f)
    
    # map chromosome arms for blockbins
    df_1['chrom_arm'] = df_1.blockbin.map(chrom_arm_dict)
    
    # get cell types in this sample
    gbm_cells = combined_df.loc[combined_df['ID'] == JOB_SAMPLE].copy()
    # remove underscore from IDs
    gbm_cells['cell_id'] = gbm_cells['cell_id'].apply(lambda x: int(x.split('_')[0]))
    # make dictionary of cell ID and cell type
    gbm_cell_dict = dict(zip(gbm_cells['cell_id'], gbm_cells['predicted_celltype']))
    
    # add cell types to df_1
    df_1['predicted_celltype'] = df_1['matched_label'].map(gbm_cell_dict)
    celltypes = df_1['predicted_celltype'].unique()
    
    # load DNA Fish locus data
    dna_loci = pd.read_csv('/resnick/groups/CaiLab/jjfox/copied_from_central/gbm_dna_analysis/25-07-28_combined_dna/human_1k_genomic_loci_final.csv')
    
    data = []
    for ct in celltypes:
        # keep celltype
        df_celltype = df_1.loc[df_1['predicted_celltype'] == ct]
        # get list of cell in this celltype
        cell_list = df_celltype.matched_label.unique()
    
        order_list = []
        min_dist_list = []
        
        for cell in cell_list:
            df_1cell = df_celltype.loc[df_celltype['matched_label'] == cell]
            # separate by barcoded and serial
            df_cell_chr = df_1cell[df_1cell['block_assigned'].notna()].copy()
            df_cell_other = df_1cell[~df_1cell['block_assigned'].notna()].copy()
        
            # separate EGFR dots
            df_cell_egfr = df_cell_other.loc[df_cell_other['name'] == target]      ######## change target here for distances ##########
            # Build KDTree from EGFR points
            egfr_points = df_cell_egfr[['x', 'y', 'z']].values
            tree = cKDTree(egfr_points)
        
            # determine order from block and bin
            # extract block number
            df_cell_chr['block_num'] = df_cell_chr['block_assigned'].str.extract(r'block_(\d+)').astype(int)
            # extract bin number
            df_cell_chr['bin_num'] = df_cell_chr['name'].str.extract(r'bin_(\d+)').astype(int)
            # create order
            df_cell_chr['order'] = ((df_cell_chr['block_num'] - 1) * 15) + df_cell_chr['bin_num']
            
            # get list of orders to check
            orders = list(df_cell_chr['order'].unique())
            # sort list
            orders = sorted(orders)
        
            # get closest EGFR for each position
            egfr_distance = []
            
            # loop through orders
            for order in orders:
                # slice the dataframe
                order_df = df_cell_chr.loc[df_cell_chr['order'] == order].copy()
            
                # Query distances to nearest EGFR point for each point in order_df
                order_points = order_df[['x', 'y', 'z']].values
                distances, _ = tree.query(order_points, k=1)
                
                # Add distances as a new column
                order_df['dist_to_nearest_egfr'] = distances
                # get minimum distance
                min_distance = order_df['dist_to_nearest_egfr'].min()
                # add distance to list
                egfr_distance.append(min_distance)
        
            # append data
            order_list.append(orders)
            min_dist_list.append(egfr_distance)
        # append data
        data_tuple = (order_list, min_dist_list)
        data.append(data_tuple)
    
    # Save as pickle
    with open(f'{target}_distance_{JOB_SAMPLE}.pickle', 'wb') as f:
        pickle.dump(data, f)
    
    with open(f'celltypes_for{target}_{JOB_SAMPLE}.pickle', 'wb') as f:
        pickle.dump(celltypes, f)
    
    # determine where to draw lines separating chromosomes
    # extract block number
    dna_loci['block_num'] = dna_loci['bin_name'].str.extract(r'block(\d+)_').astype(int)
    # extract bin number
    dna_loci['bin_num'] = dna_loci['bin_name'].str.extract(r'bin(\d+)').astype(int)
    # create order
    dna_loci['order'] = ((dna_loci['block_num'] - 1) * 15) + dna_loci['bin_num']
    
    # get list of orders where chromosomes start
    change_points = dna_loci.loc[
        dna_loci["bin_chromosome"] != dna_loci["bin_chromosome"].shift(),
        ["bin_chromosome", "order"]
    ].reset_index(drop=True)
    
    if plot_normal_v_malignant:
        ###### plot normal v malignant
        
        # --- parameters ---
        n_boot = 1000
        alpha = 0.05
        
        # --- set up plot ---
        plt.figure(figsize=(10, 6))
        
        # --- define color palette ---
        colors = palette_hex[:2]
        
        def bootstrap_ci(values, n_boot=1000, alpha=0.05):
            """Return mean, lower, and upper CI using bootstrap."""
            values = np.array(values)
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
        
        # get positions of normal vs malignant cells
        normal_list = ['Microglia', 'Pericytes', 'Neuron GABA.', 'Neuron Glut.', 'Olig. normal', 'Endothelial', 'T', 'AC']
        malig_list = ['AC.like', 'OPC.like Prolif.', 'OPC']
        positions_normal = np.where(np.isin(celltypes, normal_list))[0].tolist()
        positions_malig = np.where(np.isin(celltypes, malig_list))[0].tolist()
        
        # --- loop over all cell types --- separate normal and maligant cells
        x_normal = []
        y_normal = []
        
        x_malig = []
        y_malig = []
        
        # normal cells
        for celltyp in positions_normal:    # change these numbers depending on the cell types
            # extract data
            x_lists, y_lists = data[celltyp]
            x_normal.extend(x_lists)
            y_normal.extend(y_lists)
            
        # malig cells
        for celltyp in positions_malig:    # change these numbers depending on the cell types
            # extract data
            x_lists, y_lists = data[celltyp]
            x_malig.extend(x_lists)
            y_malig.extend(y_lists)
        
        # plot
        for xl, yl, lab, col in [
            (x_normal, y_normal, 'normal', 0),
            (x_malig, y_malig, 'malig', 1)
        ]:
            # collect all unique x values
            all_x = sorted(set(np.concatenate(xl)))
        
            # align y-values into a matrix (NaN for missing x's)
            y_matrix = np.full((len(yl), len(all_x)), np.nan)
            for k, (xs, ys) in enumerate(zip(xl, yl)):
                xs = np.array(xs)
                ys = np.array(ys)
                for j, x_val in enumerate(all_x):
                    match = np.where(xs == x_val)[0]
                    if len(match) > 0:
                        y_matrix[k, j] = ys[match[0]]
        
            # compute bootstrapped mean and CI for each x
            mean_y, lower_y, upper_y = [], [], []
            for j in range(len(all_x)):
                m, l, u = bootstrap_ci(y_matrix[:, j], n_boot, alpha)
                mean_y.append(m)
                lower_y.append(l)
                upper_y.append(u)
        
            mean_y = np.array(mean_y)
            lower_y = np.array(lower_y)
            upper_y = np.array(upper_y)
        
            color = colors[col]
            num_cells = len(yl)
            label = f"{lab} (n={num_cells})"
        
            # plot line + transparent CI
            plt.plot(all_x, mean_y, color=color, lw=1, alpha = 0.8, label=label)
            plt.fill_between(all_x, lower_y, upper_y, color=color, alpha=0.15)
        
        
        # --- add vertical chromosome lines ---
        for _, row in change_points.iterrows():
            chrom = row["bin_chromosome"]
            order_val = row["order"]
            plt.axvline(order_val, color="gray", linestyle="--", alpha=0.5)
            plt.text(
                order_val,
                plt.ylim()[1] * 0.95,
                str(chrom),
                rotation=90,
                va="top",
                ha="center",
                fontsize=8,
                color="gray",
            )
        
        # --- finalize plot ---
        plt.xlabel("bin")
        plt.ylabel(f"distance to nearest {target} dot (µm)")
        plt.title("Bootstrapped mean ± 95% CI across all cell types")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
        plt.ylim(bottom=0)
        plt.tight_layout()
        
        # --- save single combined figure ---
        save_path = f"../../../figures/Figure_2/Figure_2C/{JOB_SAMPLE}_normalvsmalignant_{target}_proximity_trace.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()