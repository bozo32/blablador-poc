#!/bin/sh
set -eu

# Long-running local-only corpus triage:
# - snapshots a file list under corpus/local
# - scans in batches against GROBID directly
# - moves scanned PDFs out of corpus/local into corpus/ok and corpus/focus

batch="${BATCH:-10}"
timeout_s="${TIMEOUT_S:-60}"
endpoints="${ENDPOINTS:-header,refs}"
grobid_url="${GROBID_URL:-http://127.0.0.1:8070}"
retries="${RETRIES:-2}"
sleep_s="${SLEEP_S:-2}"

mkdir -p data/corpus_lists
list_id="$(date -u +%Y%m%dT%H%M%SZ)__$$"
list_path="data/corpus_lists/${list_id}.txt"

bash scripts/dev/make_pdf_list.sh corpus/local >"$list_path"
total=$(python -c "print(sum(1 for _ in open('$list_path','r',encoding='utf-8')))" )

echo "list_path=${list_path}" 1>&2
echo "total=${total}" 1>&2
echo "batch=${batch}" 1>&2
echo "grobid_url=${grobid_url}" 1>&2
echo "endpoints=${endpoints}" 1>&2
echo "timeout_s=${timeout_s}" 1>&2

start=1
while [ "$start" -le "$total" ]; do
  echo "--- batch start=${start} ---" 1>&2
  out_dir=$(GROBID_URL="$grobid_url" ENDPOINTS="$endpoints" START="$start" LIMIT="$batch" TIMEOUT_S="$timeout_s" RETRIES="$retries" SLEEP_S="$sleep_s" bash scripts/dev/scan_grobid_filelist.sh "$list_path")
  bash scripts/dev/tidy_local_corpus.sh "$out_dir" 1>&2
  start=$((start + batch))
done

echo "Triage complete. Remaining inbox PDFs:" 1>&2
bash -lc "python - <<'PY'
import os
root='corpus/local'
c=0
for d,_,files in os.walk(root):
    for fn in files:
        if fn.lower().endswith('.pdf'):
            c+=1
print(c)
PY" 1>&2
