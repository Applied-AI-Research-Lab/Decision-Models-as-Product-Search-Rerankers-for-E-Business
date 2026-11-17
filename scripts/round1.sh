#!/usr/bin/env bash
set -euo pipefail
P="$(cd "$(dirname "$0")/.." && pwd)"
STAGE=${1:?usage: round1.sh prepare|pilot|full|status}
cd "$P"; mkdir -p logs

ESCI=esci_us,esci_es,esci_jp
ESJP=esci_es,esci_jp
NEW=esci_us_trainprobe,wands,esci_us_retrieval
RR8_REV=77d193c791ed757ca307ee72715aa132723da912

case "$STAGE" in
  pilot) RUN=round1_pilot; LIM="--limit 12"; BUDGET="5000"; FULLTRAIN="" ;;
  full)  RUN=round1;       LIM="";           BUDGET="5000 20000 50000"; FULLTRAIN="full" ;;
  prepare|status) RUN=round1 ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
A="--run-name $RUN"
log() { echo "logs/${RUN}_$1.log"; }

start() {
  local name=$1 cmd=$2
  if tmux has-session -t "$name" 2>/dev/null; then
    echo "!! session $name already exists (attach or kill it: tmux kill-session -t $name)"; return
  fi
  local f="logs/${RUN}_cmd_${name}.csh"
  printf '%s\n' "${cmd//; /$'\n'}" > "$f"
  tmux new-session -d -s "$name" tcsh
  tmux send-keys -t "$name" "cd $P" Enter
  tmux send-keys -t "$name" "source $f" Enter
  echo "started $name"
}

if [[ "$STAGE" == "status" ]]; then
  tmux ls 2>/dev/null | grep bp03r1 || echo "no bp03r1 sessions"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
  for f in logs/round1*_*.log; do [[ -f $f ]] && { echo "== $f"; tail -n 2 "$f"; }; done
  for r in results/round1_pilot results/round1; do
    [[ -d $r/raw ]] || continue
    echo "== $r/raw (lines per file; errors in brackets)"
    find "$r/raw" -name '*.jsonl' | sort | while read -r f; do
      n=$(wc -l < "$f"); e=$(grep -c '"error": "' "$f" || true)
      printf "%-70s %7d [%d]\n" "${f#$r/raw/}" "$n" "$e"
    done
  done
  exit 0
fi

if [[ "$STAGE" == "prepare" ]]; then
  start bp03r1-prep "source activate_project.csh && uv pip install --python .venv/bin/python snowballstemmer==3.0.1 fugashi==1.5.2 unidic-lite==1.0.8 |& tail -1 && hf download Qwen/Qwen3-Reranker-8B --revision $RR8_REV |& tail -2 && python -m bp03.data_round1 |& tee -a $(log data) && echo PREPARE-DONE"
  exit 0
fi

VLLM="setenv VLLM_USE_DEEP_GEMM 0 && python -m bp03.runners.vllm_llm"
RR="python -m bp03.runners.reranker"
MB="python -m bp03.runners.baselines --which modernbert"
X="--any-condition"

readout1() {
  echo "$VLLM --llm $1 --mode readout1 --tasks $ESCI $A $LIM |& tee -a $(log $2_main); $VLLM --llm $1 --mode readout1 --tasks $ESJP --conditions score,noul $X $A $LIM |& tee -a $(log $2_esjp); $VLLM --llm $1 --mode readout1 --tasks $NEW --conditions choice $A $LIM |& tee -a $(log $2_new)"
}

rerankers() {
  local c="$RR --size 4b --instruction ecommerce --tasks $NEW $A $LIM |& tee -a $(log gpu2_rr_4b_ecommerce)"
  for ins in default esci; do c="$c; $RR --size 4b --instruction $ins --tasks $ESCI,$NEW $A $LIM |& tee -a $(log gpu2_rr_4b_$ins)"; done
  for ins in default ecommerce esci; do c="$c; $RR --size 8b --instruction $ins --tasks $ESCI,$NEW $A $LIM |& tee -a $(log gpu2_rr_8b_$ins)"; done
  echo "$c"
}

budget() {
  local c="true" n s
  for n in $BUDGET; do for s in 0 1 2; do
    c="$c; $MB --train-n $n --seed-offset $s --tasks esci_us,esci_us_trainprobe,wands,esci_us_retrieval --conditions choice $X $A $LIM |& tee -a $(log gpu2_mb_${n}_$s)"
  done; done
  for n in $FULLTRAIN; do
    c="$c; $MB --train-n $n --seed-offset 0 --tasks esci_us,esci_us_trainprobe,wands,esci_us_retrieval --conditions choice $X $A $LIM |& tee -a $(log gpu2_mb_${n}_0)"
  done
  echo "$c"
}

start bp03r1-gpu0 "source activate_project.csh && setenv CUDA_VISIBLE_DEVICES 0 && $(readout1 qwen gpu0_qwen_readout1); source activate_clef.csh && python -m bp03.runners.clef --variant clef_flash --tasks $ESJP --conditions score,noul $X $A $LIM |& tee -a $(log gpu0_clef_flash_esjp); python -m bp03.runners.clef --variant clef_flash --tasks $NEW --conditions choice $A $LIM |& tee -a $(log gpu0_clef_flash_new); echo GPU0-DONE"

start bp03r1-gpu1 "source activate_project.csh && setenv CUDA_VISIBLE_DEVICES 1 && $(readout1 gemma gpu1_gemma_readout1); $VLLM --llm gemma --mode generative --tasks esci_us_trainprobe,wands $A $LIM |& tee -a $(log gpu1_gemma_gen); $VLLM --llm qwen --mode generative --tasks esci_us_trainprobe,wands $A $LIM |& tee -a $(log gpu1_qwen_gen); echo GPU1-DONE"

start bp03r1-gpu2 "source activate_project.csh && setenv CUDA_VISIBLE_DEVICES 2 && $(rerankers); $(budget); source activate_clef.csh && python -m bp03.runners.clef --variant clef --tasks $ESJP --conditions score,noul $X $A $LIM |& tee -a $(log gpu2_clef_esjp); python -m bp03.runners.clef --variant clef --tasks $NEW --conditions choice $A $LIM |& tee -a $(log gpu2_clef_new); echo GPU2-DONE"

start bp03r1-gpu3 "source activate_pplx.csh && setenv CUDA_VISIBLE_DEVICES 3 && python -m bp03.runners.pplx --tasks $ESJP --conditions score,noul $X $A $LIM |& tee -a $(log gpu3_pplx_esjp); python -m bp03.runners.pplx --tasks $NEW --conditions choice $A $LIM |& tee -a $(log gpu3_pplx_new); echo GPU3-DONE"

api() {
  echo "source activate_project.csh && python -m bp03.runners.api --provider $1 --tasks esci_us --conditions repeat1 $A $LIM |& tee -a $(log api_$1_drift); python -m bp03.runners.api --provider $1 --tasks $ESJP --conditions score,noul $X $A $LIM |& tee -a $(log api_$1_esjp); python -m bp03.runners.api --provider $1 --tasks $NEW --conditions choice $A $LIM |& tee -a $(log api_$1_new); echo API-$1-DONE"
}
start bp03r1-jev "$(api jev)"
start bp03r1-pplxapi "$(api pplx_api)"

start bp03r1-finish "bash scripts/round1_finish.sh $RUN |& tee -a $(log finish)"

echo "All sessions started. Monitor with: bash scripts/round1.sh status"
