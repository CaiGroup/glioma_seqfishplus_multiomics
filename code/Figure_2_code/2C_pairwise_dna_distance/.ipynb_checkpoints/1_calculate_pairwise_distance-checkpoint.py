import glob
import numpy as np
import pandas as pd
import os
import re
import pickle
import sys
sys.setrecursionlimit(50000)
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy import stats
from collections import defaultdict
from tqdm import tqdm, trange
import time

import torch

# Set GPU device
device = torch.device('cuda:0')

# Start timing
start_time = time.time()

# SLURM array index
JOB_ID = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
# Process only 1 sample per array job
samps = ['2543',
 '2683',
 '3097_1',
 '3825',
 '3883',
 '4494_2',
 '4523',
 '4555',
 '4668',
 '5121',
 '5201',
 '5538_1',
 '5597']
JOB_SAMPLE = samps[JOB_ID]

# Load all loci
with open('/resnick/groups/mthomson/ci2_data/CaiLab/gbm/dna_distance_data/used_loci.pickle', 'rb') as f:
    used_loci = pickle.load(f)

print(f"Processing sample {JOB_SAMPLE} for {len(used_loci)} loci")
print(f"Loci to process: {used_loci[:10]}... (showing first 10)")
print()

##### LOAD DATA ONCE #####
print(f"Loading data for sample {JOB_SAMPLE}...")
load_start = time.time()

# look at specific sample
path = '/resnick/groups/mthomson/ci2_data/CaiLab/gbm/Data/DNA_data'
# get all files
files = glob.glob(f"{path}/*assigned.csv")
# only keep JOB_SAMPLE
patient_id = JOB_SAMPLE
if '_' in patient_id:
    patient_id = patient_id.split('_')[0]
files = [i for i in files if patient_id in i]
# load data
df_1 = pd.read_csv(files[0], low_memory = False)
print(f"Data loaded in {time.time() - load_start:.2f}s")

##### PREPROCESS DATA ONCE #####
print("Preprocessing data...")
preprocess_start = time.time()

# fix chromosomes
# make dictionary
probe_df = pd.read_csv("/resnick/groups/mthomson/ci2_data/CaiLab/gbm/dna_distance_data/24-12-19_human_genome_DNA_seqFISH_final_probes.csv")
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
with open('/resnick/groups/mthomson/ci2_data/CaiLab/gbm/dna_distance_data/chrom_arm_dict.pkl', 'rb') as f:
    chrom_arm_dict = pickle.load(f)

# map chromosome arms for blockbins
df_1['chrom_arm'] = df_1.blockbin.map(chrom_arm_dict)

# load cell type data
celltype_df = pd.read_csv('/resnick/groups/mthomson/ci2_data/CaiLab/gbm/dna_distance_data/V2_24_patients_Obj_mapped.csv', low_memory=False)
# replace . with _ in column names
celltype_df.columns = [
    col.replace('.', '_') for col in celltype_df.columns
]
# Rename column for cell ID
celltype_df.rename(columns={'Unnamed: 0': 'cell_id'}, inplace=True)
# get cell types in this sample
gbm_cells = celltype_df.loc[celltype_df['ID'] == JOB_SAMPLE].copy()
# remove underscore from IDs
gbm_cells['cell_id'] = gbm_cells['cell_id'].apply(lambda x: int(x.split('_')[0]))
# make dictionary of cell ID and cell type
gbm_cell_dict = dict(zip(gbm_cells['cell_id'], gbm_cells['predicted_celltype']))                
# add cell types to df_1
df_1['predicted_celltype'] = df_1['matched_label'].map(gbm_cell_dict)
celltypes = df_1['predicted_celltype'].unique()

# filter out cells with less than 250 unqiue bins
# Count unique blockbin values per cell (matched_label)
blockbin_counts = df_1.groupby('matched_label')['blockbin'].nunique()
# Get cells with more than 250 unique blockbin values
cells_to_keep = blockbin_counts[blockbin_counts > 250].index
# Filter the dataframe
df_1 = df_1[df_1['matched_label'].isin(cells_to_keep)]

print(f"Preprocessing completed in {time.time() - preprocess_start:.2f}s")
print(f"Found {len(celltypes)} cell types, {len(df_1['matched_label'].unique())} total cells")
print()

##### PROCESS ALL CELLS FOR ALL LOCI #####
print("Processing all cells for all loci...")
processing_start = time.time()

# Initialize data structures for each locus
loci_data = {locus: [] for locus in used_loci}
loci_cell_id_data = {locus: [] for locus in used_loci}

# Process by cell type
for ct_idx, ct in enumerate(celltypes):
    print(f"\nProcessing cell type {ct_idx+1}/{len(celltypes)}: {ct}")
    
    # keep celltype
    df_celltype = df_1.loc[df_1['predicted_celltype'] == ct]
    # get list of cell in this celltype
    cell_list = df_celltype.matched_label.unique()
    print(f"  Found {len(cell_list)} cells")
    
    # Initialize data structures for this cell type for each locus
    for locus in used_loci:
        loci_data[locus].append([[], []])  # [order_list, min_dist_list]
    
    # Process each cell ONCE
    for cell in tqdm(cell_list, desc=f"  Cells in {ct}"):
        df_1cell = df_celltype.loc[df_celltype['matched_label'] == cell]
        # separate by barcoded and serial
        df_cell_chr = df_1cell[df_1cell['block_assigned'].notna()].copy()
        
        # determine order from block and bin (ONCE per cell)
        df_cell_chr['block_num'] = df_cell_chr['block_assigned'].str.extract(r'block_(\d+)').astype(int)
        df_cell_chr['bin_num'] = df_cell_chr['name'].str.extract(r'bin_(\d+)').astype(int)
        df_cell_chr['order'] = ((df_cell_chr['block_num'] - 1) * 15) + df_cell_chr['bin_num']
        
        # get list of orders to check (ONCE per cell)
        orders = list(df_cell_chr['order'].unique())
        orders = sorted(orders)
        
        # Move all cell points to GPU once
        all_points = torch.tensor(df_cell_chr[['x', 'y', 'z']].values, dtype=torch.float64, device=device)
        all_orders_array = df_cell_chr['order'].values
        
        # Now process each locus for this cell
        for locus in used_loci:
            # separate target locus dots
            df_cell_target = df_cell_chr.loc[df_cell_chr['blockbin'] == locus]
            
            # Check if there are any target points for this cell
            if len(df_cell_target) == 0:
                # No target points - return inf for all orders (matching CPU cKDTree behavior)
                egfr_distance = []
                for order in orders:
                    order_mask = all_orders_array == order
                    if order_mask.sum() == 0:
                        egfr_distance.append(np.nan)
                    else:
                        egfr_distance.append(np.inf)
            else:
                # Get target points and move to GPU (use float64 to match CPU precision)
                target_points = torch.tensor(df_cell_target[['x', 'y', 'z']].values, dtype=torch.float64, device=device)
                
                # Compute ALL distances at once on GPU
                all_dists = torch.cdist(all_points, target_points)
                min_dists_per_point = torch.min(all_dists, dim=1)[0]
                
                # Move results back to CPU for grouping
                min_dists_cpu = min_dists_per_point.cpu().numpy()
                
                # Group by order and take minimum distance for each order
                egfr_distance = []
                for order in orders:
                    order_mask = all_orders_array == order
                    if order_mask.sum() == 0:
                        egfr_distance.append(np.nan)
                    else:
                        min_dist_for_order = min_dists_cpu[order_mask].min()
                        egfr_distance.append(min_dist_for_order)
            
            # Append data for this locus
            loci_data[locus][-1][0].append(orders)  # order_list
            loci_data[locus][-1][1].append(egfr_distance)  # min_dist_list

        # FREE GPU MEMORY after processing all loci for this cell
        del all_points
        torch.cuda.empty_cache()
    
    # Append cell IDs for each locus (same for all loci)
    for locus in used_loci:
        loci_cell_id_data[locus].append(list(cell_list))

print(f"\nProcessing completed in {time.time() - processing_start:.2f}s")

##### SAVE RESULTS FOR EACH LOCUS #####
print("\nSaving results for each locus...")
save_start = time.time()

for locus in tqdm(used_loci, desc="Saving loci"):
    # Convert data structure to match original format
    data = []
    for ct_idx in range(len(celltypes)):
        order_list = loci_data[locus][ct_idx][0]
        min_dist_list = loci_data[locus][ct_idx][1]
        data_tuple = (order_list, min_dist_list)
        data.append(data_tuple)
    
    # Save as pickle
    with open(f'/resnick/groups/mthomson/ci2_data/CaiLab/gbm/dna_distance_data/proximity_data_gpu/{locus}_distance_{JOB_SAMPLE}.pickle', 'wb') as f:
        pickle.dump(data, f)
    
    with open(f'/resnick/groups/mthomson/ci2_data/CaiLab/gbm/dna_distance_data/proximity_data_gpu/celltypes_for{locus}_{JOB_SAMPLE}.pickle', 'wb') as f:
        pickle.dump(celltypes, f)

    with open(f'/resnick/groups/mthomson/ci2_data/CaiLab/gbm/dna_distance_data/proximity_data_gpu/cellIDs_for{locus}_{JOB_SAMPLE}.pickle', 'wb') as f:
        pickle.dump(loci_cell_id_data[locus], f)

print(f"Saving completed in {time.time() - save_start:.2f}s")

# End timing
end_time = time.time()
elapsed_time = end_time - start_time

print("\n" + "="*60)
print("PROCESSING COMPLETE!")
print("="*60)
print(f"Sample: {JOB_SAMPLE}")
print(f"Number of loci processed: {len(used_loci)}")
print(f"Number of cells processed: {len(df_1['matched_label'].unique())}")
print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
print(f"Average time per locus: {elapsed_time/len(used_loci):.2f} seconds")
print("="*60)
