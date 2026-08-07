"""fulcrum CLI: `fulcrum metrica 1 --moment 300` or `python -m fulcrum skillcorner 2017461`."""
from __future__ import annotations
import argparse
from .pipeline import analyze


def main(argv=None):
    ap = argparse.ArgumentParser("fulcrum", description="Topological hole-finder for football tracking -> annotated clips.")
    ap.add_argument("source", choices=["metrica", "skillcorner"])
    ap.add_argument("match", type=int, help="match id")
    ap.add_argument("--moment", type=float, default=None, help="seconds into the match; omit to auto-detect chances")
    ap.add_argument("--out", default="fulcrum_out", help="output directory")
    ap.add_argument("--top", type=int, default=3, help="number of ranked holes to show")
    ap.add_argument("--static", action="store_true", help="render a PNG frame instead of an animated clip")
    ap.add_argument("--max-chances", type=int, default=6, help="how many auto-detected chances to render")
    ap.add_argument("--fmt", default="gif", choices=["gif", "mp4"], help="clip format")
    a = ap.parse_args(argv)
    for r in analyze(a.source, a.match, moment_s=a.moment, out_dir=a.out, top=a.top,
                     animate=not a.static, max_chances=a.max_chances, fmt=a.fmt):
        if "error" in r:
            print(f"  frame {r['fid']}: skipped ({r['error']})")
        else:
            print(f"  frame {r['fid']}: {len(r['holes'])} holes, top score {r['holes'][0]['score'] if r['holes'] else '-'} -> {r['out']}")


if __name__ == "__main__":
    main()
