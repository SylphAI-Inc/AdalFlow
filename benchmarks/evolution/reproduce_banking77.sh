#!/usr/bin/env bash
# Reproduce the Banking77 comparison from "Lineage Is Memory" (Fig. 2):
#   parent-full-first            build on the best parent, no carry-facts   (paper: 0.372)
#   parent-full-first + carry    the same plus CARRY_FACTS                  (paper: 0.856)
#   GEPA 0.0.7, minibatch 3      at the same task-call budget               (paper: 0.664)
# Held-out score of the best search-set candidate, gpt-4.1-mini in all roles,
# 12 intents, 8 evaluations. About $1 at REPEATS=10 (paper used 10).
#
#   OPENAI_API_KEY=... bash benchmarks/evolution/reproduce_banking77.sh
#   REPEATS=3 OUT=runs/repro bash benchmarks/evolution/reproduce_banking77.sh
#   CHECK=1 bash benchmarks/evolution/reproduce_banking77.sh   # fake models, no API
set -euo pipefail
cd "$(dirname "$0")/../.."
REPEATS=${REPEATS:-10}
OUT=${OUT:-runs/repro_banking77}
export PUBLIC_DATA_DIR=${PUBLIC_DATA_DIR:-data/public}
D=$PUBLIC_DATA_DIR/banking77
URL=https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data

mkdir -p "$D" "$OUT"
for f in train test; do
  [ -s "$D/$f.csv" ] || curl -sfL -o "$D/$f.csv" "$URL/$f.csv"
done
(cd "$D" && sha256sum -c --quiet - <<'EOF'
b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b  train.csv
d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d  test.csv
EOF
)

PA=benchmarks/evolution/prompt_anchoring.py
if [ -n "${CHECK:-}" ]; then
  python $PA --check --task banking77 --repeats 1 --cells parent-full-first --out "$OUT/check" >/dev/null
  python $PA --check --task banking77 --repeats 1 --cells parent-full-first --carry-facts --out "$OUT/check" >/dev/null
  echo "check ok (no API calls); GEPA needs a real model and is skipped"
  exit 0
fi
: "${OPENAI_API_KEY:?set OPENAI_API_KEY}"
MINI="--provider openai --model gpt-4.1-mini"
python $PA $MINI --task banking77 --repeats "$REPEATS" --workers 8 --cells parent-full-first --out "$OUT/plain"
python $PA $MINI --task banking77 --repeats "$REPEATS" --workers 8 --cells parent-full-first --carry-facts --out "$OUT/carry"
python benchmarks/evolution/gepa_baseline.py $MINI --task banking77 --repeats "$REPEATS" --out "$OUT/gepa"

python - "$OUT" <<'EOF'
import json, statistics as st, sys
from pathlib import Path
out = Path(sys.argv[1])
arms = [("parent-first", out / "plain/summary.json", "parent-full-first", 0.372),
        ("parent-first + carry-facts", out / "carry/summary.json", "parent-full-first", 0.856),
        ("GEPA 0.0.7 mb3", out / "gepa/summary-mb3.json", None, 0.664)]
print("\n| arm | held-out best (mean) | runs | tokens/run (median) | paper |\n|---|---|---|---|---|")
for name, path, key, paper in arms:
    rows = [r for r in json.loads(path.read_text()) if key is None or r["key"] == key]
    hb = [r["holdout_best"] for r in rows]
    tok = st.median(r["proposer_tokens"] + r["task_tokens"] for r in rows)
    print(f"| {name} | {st.mean(hb):.3f} | {len(hb)} | {tok:.0f} | {paper:.3f} |")
EOF
