import argparse
import h5py
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file1', type=str, help="Path to the first dataset file")
    parser.add_argument('file2', type=str, help="Path to the second dataset file")
    parser.add_argument('--output_file', type=str, default='output.hdf5', help="Path for the stitched output file")
    parser.add_argument('--maxlen', type=int, default=2000000, help="Maximum number of samples in the output file")
    args = parser.parse_args()

    # Load input files
    hfile1 = h5py.File(args.file1, 'r')
    hfile2 = h5py.File(args.file2, 'r')
    outf = h5py.File(args.output_file, 'w')

    # Define keys to stitch
    all_keys = [
        'observations', 'next_observations', 'actions', 'rewards',
        'terminals', 'timeouts', 'infos/action_log_probs'
    ]
    
    # Metadata keys to copy
    metadata_keys = [
        'metadata/algorithm', 'metadata/policy/nonlinearity',
        'metadata/policy/output_distribution'
    ]

    # Find the end of the last trajectory in file1
    if 'terminals' in hfile1 and 'timeouts' in hfile1:
        terms = hfile1['terminals'][:]
        tos = hfile1['timeouts'][:]
        last_term = 0
        for i in range(len(terms) - 1, -1, -1):
            if terms[i] or tos[i]:
                last_term = i
                break
        N = last_term + 1
    else:
        N = len(hfile1['observations'])

    # Stitch the datasets
    for k in all_keys:
        if k in hfile1 and k in hfile2:
            d1 = hfile1[k][:N]
            d2 = hfile2[k][:]
            combined = np.concatenate([d1, d2], axis=0)[:args.maxlen]
            print(f"{k}: {combined.shape}")
            outf.create_dataset(k, data=combined, compression='gzip')
        elif k in hfile1:
            d1 = hfile1[k][:N][:args.maxlen]
            print(f"{k}: {d1.shape}")
            outf.create_dataset(k, data=d1, compression='gzip')
        elif k in hfile2:
            d2 = hfile2[k][:args.maxlen]
            print(f"{k}: {d2.shape}")
            outf.create_dataset(k, data=d2, compression='gzip')
        else:
            print(f"Key {k} not found in either file, skipping.")

    # Add metadata
    for mk in metadata_keys:
        if mk in hfile1.attrs:
            outf.attrs[mk] = hfile1.attrs[mk]
        elif mk in hfile2.attrs:
            outf.attrs[mk] = hfile2.attrs[mk]
        else:
            print(f"Metadata key {mk} not found in either file, skipping.")

    # Close files
    hfile1.close()
    hfile2.close()
    outf.close()

    print(f"Stitched dataset saved to {args.output_file}")
