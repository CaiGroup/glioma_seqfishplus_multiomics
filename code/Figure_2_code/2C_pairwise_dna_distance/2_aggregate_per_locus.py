import os
import pickle
import numpy as np
import glob
from multiprocessing import Pool
from functools import partial

# ========================
# CONFIGURATION
# ========================
OVERWRITE = True  # Set to True to overwrite existing files

# define samples here
samps = ['5538_1', '5538_2']

JOB_ID = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))

JOB_SAMPLE = samps[JOB_ID]

# ========================
# helper: compute mean trace
# ========================
def compute_mean_trace_fixed(data, positions, x_axis=None):
    xl, yl = [], []

    for celltyp in positions:
        x_lists, y_lists = data[celltyp]
        xl.extend(x_lists)
        yl.extend(y_lists)

    if x_axis is None:
        all_x = sorted(set(np.concatenate(xl)))
    else:
        all_x = x_axis

    y_matrix = np.full((len(yl), len(all_x)), np.nan)
    for k, (xs, ys) in enumerate(zip(xl, yl)):
        xs = np.array(xs)
        ys = np.array(ys)
        for j, x_val in enumerate(all_x):
            match = np.where(xs == x_val)[0]
            if len(match) > 0:
                y_matrix[k, j] = ys[match[0]]

    mean_y = np.empty(len(all_x))
    mean_y[:] = np.nan

    for j in range(len(all_x)):
        vals = y_matrix[:, j]
        vals = vals[np.isfinite(vals) & (vals != 0)]
        if len(vals) > 0:
            mean_y[j] = np.mean(vals)

    return all_x, mean_y


# ========================
# process single locus
# ========================
def process_locus(job_id, used_loci, overwrite=False):
    """Process a single locus given its job_id"""
    
    targ = used_loci[job_id]
    
    # ========================
    # check if the file already exists
    # ========================
    out_file = f"/dna_distance_data/per_locus_results_gpu/{job_id:04d}_{targ}_{JOB_SAMPLE}.pickle"
    
    if os.path.exists(out_file) and not overwrite:
        return f"[JOB {job_id}] skipped {targ} (already exists)"
    
    # ========================
    # parameters
    # ========================
    normal_list = [
        'Microglia', 'Pericytes', 'Neuron GABA.', 'Neuron Glut.',
        'Olig. normal', 'Endothelial', 'T', 'AC'
    ]
    malig_list = ['AC.like', 'OPC.like Prolif.', 'OPC']
    
    # ========================
    # locate files
    # ========================
    path = '/dna_distance_data/proximity_data_gpu/'
    # get list of pickle files from path
    files = glob.glob(f"{path}/*{JOB_SAMPLE}.pickle")
    # split into distance and cell type
    dist_files = [i for i in files if f'_distance_{JOB_SAMPLE}' in i]
    celltype_files = [i for i in files if 'celltypes' in i]
    cellID_files = [i for i in files if 'cellIDs' in i]
    
    match_data_file, = [x for x in dist_files if f"{targ}_" in x]
    match_celltype_file, = [x for x in celltype_files if f"{targ}_" in x]
    match_cellID_file, = [x for x in cellID_files if f"{targ}_" in x]
    
    # ========================
    # load data
    # ========================
    with open(match_data_file, "rb") as f:
        data = pickle.load(f)
    
    with open(match_celltype_file, "rb") as f:
        celltypes = pickle.load(f)

    with open(match_cellID_file, "rb") as f:
        cellIDs = pickle.load(f)
    
    # ========================
    # compute normal & malignant
    # ========================
    pos_normal = np.where(np.isin(celltypes, normal_list))[0].tolist()
    pos_malig  = np.where(np.isin(celltypes, malig_list))[0].tolist()
    
    # First, get all unique x values from BOTH cell types
    all_x_normal, _ = compute_mean_trace_fixed(data, pos_normal)
    all_x_malig, _  = compute_mean_trace_fixed(data, pos_malig)
    all_x_union = sorted(set(all_x_normal).union(set(all_x_malig)))
    
    # Now compute means using the same x-axis for both
    x_axis, mean_normal = compute_mean_trace_fixed(data, pos_normal, all_x_union)
    _,      mean_malig  = compute_mean_trace_fixed(data, pos_malig, all_x_union)
    
    # ========================
    # save output
    # ========================
    out = {
        "locus": targ,
        "x_axis": x_axis,
        "mean_normal": mean_normal,
        "mean_malig": mean_malig,
    }
    
    os.makedirs("/dna_distance_data/per_locus_results_gpu", exist_ok=True)
    
    with open(out_file, "wb") as f:
        pickle.dump(out, f)
    
    status = "overwritten" if (os.path.exists(out_file) and overwrite) else "finished"
    return f"[JOB {job_id}] {status} {targ}"


# ========================
# main execution
# ========================
if __name__ == '__main__':
    # ========================
    # load locus list
    # ========================
    with open('/dna_distance_data/used_loci.pickle', "rb") as f:
        used_loci = pickle.load(f)
    
    print(f"Total loci to process: {len(used_loci)}")
    print(f"Overwrite mode: {'ENABLED' if OVERWRITE else 'DISABLED'}")
    
    # ========================
    # determine number of workers
    # ========================
    n_workers = int(os.getenv("SLURM_CPUS_PER_TASK", 32))
    print(f"Using {n_workers} workers")
    
    # ========================
    # create list of all job IDs
    # ========================
    job_ids = list(range(len(used_loci)))
    
    # ========================
    # process in parallel
    # ========================
    process_func = partial(process_locus, used_loci=used_loci, overwrite=OVERWRITE)
    
    completed = 0
    skipped = 0
    
    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(process_func, job_ids):
            if "skipped" in result:
                skipped += 1
            else:
                completed += 1
            
            total_done = completed + skipped
            if total_done % 50 == 0:  # Print progress every 50 loci
                print(f"Progress: {total_done}/{len(used_loci)} loci processed ({completed} new, {skipped} skipped)")
    
    print(f"\n=== COMPLETE ===")
    print(f"Total processed: {len(used_loci)}")
    print(f"Newly computed: {completed}")
    print(f"Skipped (already existed): {skipped}")