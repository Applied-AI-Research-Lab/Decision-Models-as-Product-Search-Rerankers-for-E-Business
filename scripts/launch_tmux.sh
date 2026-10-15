#!/usr/bin/env bash
set -euo pipefail
P="$(cd "$(dirname "$0")/.." && pwd)"
STAGE=${1:?usage: launch_tmux.sh prepare|pilot|full|analyze}
cd "$P"; mkdir -p logs

case "$STAGE" in
  pilot) ARGS="--limit 200 --conditions choice,score,noul --run-name pilot" ;;
  full)  ARGS="--run-name main" ;;
  prepare|analyze) ARGS="" ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac

start() {
  local name=$1 cmd=$2
  if tmux has-session -t "$name" 2>/dev/null; then
    echo "!! session $name already exists (attach or kill it: tmux kill-session -t $name)"; return
  fi
  tmux new-session -d -s "$name" tcsh
  tmux send-keys -t "$name" "cd $P" Enter
  tmux send-keys -t "$name" "$cmd" Enter
  echo "started $name"
}

log() { echo "logs/${STAGE}_$1.log"; }

if [[ "$STAGE" == "prepare" ]]; then
  start bp03-prep "source activate_project.csh && python -m bp03.data |& tee -a $(log data) && python -m bp03.records |& tee -a $(log records) && echo PREPARE-DONE"
  exit 0
fi
if [[ "$STAGE" == "analyze" ]]; then
  start bp03-analyze "source activate_project.csh && python -m bp03.analyze --run-name main |& tee -a $(log analyze) && echo ANALYZE-DONE"
  exit 0
fi

QWEN="setenv VLLM_USE_DEEP_GEMM 0 && python -m bp03.runners.vllm_llm --llm qwen"

start bp03-gpu0 "source activate_clef.csh && setenv CUDA_VISIBLE_DEVICES 0 && python -m bp03.runners.clef --variant clef $ARGS |& tee -a $(log gpu0_clef); source activate_project.csh && $QWEN --mode readout --shard 0,1,2,3/16 $ARGS |& tee -a $(log gpu0_qwen_readout); echo GPU0-DONE"

start bp03-gpu1 "source activate_project.csh && setenv CUDA_VISIBLE_DEVICES 1 && $QWEN --mode readout --shard 4,5,6,7,8,9/16 $ARGS |& tee -a $(log gpu1_qwen_readout); $QWEN --mode generative $ARGS |& tee -a $(log gpu1_qwen_gen); echo GPU1-DONE"

start bp03-gpu2 "source activate_project.csh && setenv CUDA_VISIBLE_DEVICES 2 && python -m bp03.runners.vllm_llm --llm gemma --mode readout $ARGS |& tee -a $(log gpu2_gemma_readout); python -m bp03.runners.vllm_llm --llm gemma --mode generative $ARGS |& tee -a $(log gpu2_gemma_gen); python -m bp03.runners.baselines --which modernbert $ARGS |& tee -a $(log gpu2_modernbert); $QWEN --mode readout --shard 10,11,12/16 $ARGS |& tee -a $(log gpu2_qwen_readout); echo GPU2-DONE"

start bp03-gpu3 "source activate_pplx.csh && setenv CUDA_VISIBLE_DEVICES 3 && python -m bp03.runners.pplx $ARGS |& tee -a $(log gpu3_pplx); source activate_project.csh && python -m bp03.runners.reranker --backend vllm $ARGS |& tee -a $(log gpu3_reranker); source activate_clef.csh && python -m bp03.runners.clef --variant clef_flash $ARGS |& tee -a $(log gpu3_clef_flash); source activate_project.csh && $QWEN --mode readout --shard 13,14,15/16 $ARGS |& tee -a $(log gpu3_qwen_readout); echo GPU3-DONE"

start bp03-cpu "source activate_project.csh && python -m bp03.runners.api --provider jev $ARGS |& tee -a $(log cpu_jev); echo CPU-DONE"
start bp03-pplxapi "source activate_project.csh && python -m bp03.runners.api --provider pplx_api $ARGS |& tee -a $(log cpu_pplx_api); echo PPLXAPI-DONE"

echo "All sessions started."
