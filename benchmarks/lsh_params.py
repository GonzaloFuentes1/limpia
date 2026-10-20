"""Print the (bands, rows) chosen for a few thresholds and their S-curve values.

Usage:
    python benchmarks/lsh_params.py --num-perm 128
"""

from __future__ import annotations

import argparse

from limpia.dedup import collision_probability, optimal_bands_rows
from limpia.dedup.lsh import false_negative_area, false_positive_area


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.6, 0.7, 0.8])
    args = parser.parse_args()
    print("| t | (b, r) | P(t - 0.1) | P(t) | P(t + 0.1) | FP area | FN area |")
    print("|---|---|---|---|---|---|---|")
    for t in args.thresholds:
        b, r = optimal_bands_rows(t, args.num_perm)
        p = [float(collision_probability(s, b, r)) for s in (t - 0.1, t, t + 0.1)]
        fp, fn = false_positive_area(t, b, r), false_negative_area(t, b, r)
        print(
            f"| {t:.1f} | ({b}, {r}) | {p[0]:.3f} | {p[1]:.3f} | {p[2]:.3f} | {fp:.4f} | {fn:.4f} |"
        )


if __name__ == "__main__":
    main()
