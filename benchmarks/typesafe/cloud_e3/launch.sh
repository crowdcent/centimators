#!/bin/sh
# Wait for Jev train-set features, then upload and start E3 on CrowdCent Cloud.
cd "$(dirname "$0")/.."
while [ "$(ls results/features/*__train_jev.parquet 2>/dev/null | wc -l)" -lt 4 ]; do sleep 60; done
for t in onion sst2 agnews fintweets; do
  cp results/features/${t}__train_jev.parquet cloud_e3/
  cp results/abl/${t}__jev_zeroshot__n0__s0.parquet cloud_e3/${t}__test_jev.parquet
done
cp run_stack.py tasks.py metrics.py cloud_e3/
cd cloud_e3
export CROWDCENT_API_KEY=PROXY_CROWDCENT_KEY
C="uvx --from crowdcent-challenge@latest crowdcent"
$C cloud save _FjMk0Q1lM8 run_stack.py tasks.py metrics.py *__train_jev.parquet *__test_jev.parquet --base-version 5 > save.log 2>&1
$C cloud run _FjMk0Q1lM8 --entrypoint run_stack.py --envelope gpu_s --time-limit 60 --idempotency-key jev-bench-e3-v1 > run.log 2>&1
