from bioemu.path_guidance import main as path_guidance
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sequence", type=str, required=True, help="Amino acid sequence of the protein.")
parser.add_argument("--init_npz", type=str, required=True, help="Path to initial structure NPZ (keys: pos, node_orientations).")
parser.add_argument("--feat_ref_npz", type=str, required=True, help="Path to pre-computed reference torsion features NPZ (key: feat_ref).")
parser.add_argument("--save_dir", type=str, default="path_guidance_results")
parser.add_argument("--gamma", type=float, default=0.1)
parser.add_argument("--N", type=int, default=100)
parser.add_argument("--n_calls", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--seed", type=int, default=None)
args = parser.parse_args()

path_guidance(
    sequence=args.sequence,
    init_npz=args.init_npz,
    feat_ref_npz=args.feat_ref_npz,
    N=args.N,
    eps_t=1e-3,
    max_t=0.98,
    method="euler",
    save_dir=args.save_dir,
    gamma=args.gamma,
    batch_size=args.batch_size,
    n_calls=args.n_calls,
    seed=args.seed,
)
