#!/usr/bin/env bash
RUN=${1:-round1}
P="$(cd "$(dirname "$0")/.." && pwd)"
cd "$P"
sleep 120
idle=0
while [ $idle -lt 4 ]; do
  if pgrep -u "$USER" -f "[p]ython -m bp03.runners" >/dev/null; then idle=0; else idle=$((idle+1)); fi
  sleep 30
done
echo "runners finished $(date)"
NEW=esci_us_trainprobe,wands,esci_us_retrieval
LIM=""; [[ "$RUN" == "round1_pilot" ]] && LIM="--limit 12"
for prov in pplx_api jev; do
  tcsh -c "source activate_project.csh && setenv PPLX_CONCURRENCY 4 && setenv JEV_CONCURRENCY 4 && python -m bp03.runners.api --provider $prov --tasks esci_us --conditions repeat1 --run-name $RUN $LIM && python -m bp03.runners.api --provider $prov --tasks esci_es,esci_jp --conditions score,noul --any-condition --run-name $RUN $LIM && python -m bp03.runners.api --provider $prov --tasks $NEW --conditions choice --run-name $RUN $LIM" 2>&1 | tee -a logs/${RUN}_api_resume.log
done
bash scripts/round1.sh status > logs/${RUN}_status_final.txt
NB=10000; [[ "$RUN" == "round1_pilot" ]] && NB=200
tcsh -c "source activate_project.csh && python -m bp03.analyze_round1 --run-name $RUN --n-boot $NB --check-main" 2>&1 | tee -a logs/${RUN}_analyze.log
tar czf ${RUN}-results.tar.gz results/$RUN data/tasks/manifest_round1.json data/tasks/esci_us_trainprobe data/tasks/wands data/tasks/esci_us_retrieval data/records/esci_us_trainprobe data/records/wands data/records/esci_us_retrieval logs/${RUN}_*.log
echo "FINISH-DONE $(date)  ->  $P/${RUN}-results.tar.gz"
