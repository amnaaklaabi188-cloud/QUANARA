"""Build the model matrix for one prediction scope and mode with the application's own feature code.

    python features/build_features.py --scope international_total --mode whatif
    python features/build_features.py --scope international_market --mode whatif --out data/processed/mkt_whatif.csv

The matrix is derived from the restricted competition files, so it is written under data/processed/ (git-ignored)
unless --out is given. The fingerprint printed at the end matches the one the ML Training Studio records.
"""
import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from quanara import config  # noqa: E402
from quanara.feature_facts import matrix_columns, matrix_fingerprint, training_matrix  # noqa: E402
from quanara.features import available_features, default_features  # noqa: E402
from quanara.integration import get_analytic  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=list(config.SCOPES), default="international_total")
    ap.add_argument("--mode", choices=list(config.MODES), default="whatif")
    ap.add_argument("--all-features", action="store_true", help="every feature usable in this mode (default: the recommended set)")
    ap.add_argument("--out")
    args = ap.parse_args()
    tr = training_matrix(get_analytic(), args.scope, args.mode)
    feats = available_features(args.scope, args.mode) if args.all_features else default_features(args.scope, args.mode)
    cols = matrix_columns(args.scope, feats)
    mx = tr[cols]
    out = Path(args.out) if args.out else ROOT / "data" / "processed" / f"model_matrix_{args.scope}_{args.mode}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    mx.to_csv(out, index=False)
    print(f"{config.SCOPES[args.scope]} · {config.MODES[args.mode]}\n{len(mx):,} rows × {len(feats)} features · "
          f"{mx['Date'].min():%Y-%m-%d} → {mx['Date'].max():%Y-%m-%d} · fingerprint {matrix_fingerprint(mx)}\nwrote {out}")
