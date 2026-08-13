"""Layer-enforcement test (architecture/SERVICES.md): products & applications compose SERVICES; they must NOT import
the encoder / world model directly. Intent is not architecture — the import graph is. This test FAILS the build on
a violation."""
import os, re, glob

FORBIDDEN_IN_UPPER = [r"^\s*import\s+worldmodel", r"from\s+worldmodel\s+import",
                      r"from\s+fulcrum\s+import\s+model\b", r"from\s+fulcrum\.model\s+import", r"import\s+torch\b"]
HERE = os.path.dirname(__file__)


def _violations(pattern_files):
    bad = []
    for f in pattern_files:
        src = open(f).read()
        for pat in FORBIDDEN_IN_UPPER:
            if re.search(pat, src, re.M):
                bad.append((os.path.basename(f), pat))
    return bad


def test_products_and_applications_dont_touch_the_encoder():
    upper = glob.glob(os.path.join(HERE, "..", "products", "*.py")) + glob.glob(os.path.join(HERE, "..", "applications", "*.py"))
    upper = [f for f in upper if not f.endswith("__init__.py")]
    bad = _violations(upper)
    assert not bad, f"LAYER VIOLATION — product/application imports the encoder/world-model directly: {bad}"
    print(f"test_layers: PASS — {len(upper)} product/application modules compose services only (no direct encoder/torch import)")


if __name__ == "__main__":
    test_products_and_applications_dont_touch_the_encoder()
