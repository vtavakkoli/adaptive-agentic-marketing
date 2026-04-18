#!/usr/bin/env bash
set -euo pipefail

python -m src.data.prepare
python -m src.training.train_xgboost

if [[ "${ENABLE_PPO:-1}" == "1" ]]; then
  python -m src.rl.train_ppo \
    --train-path data/processed/train.csv \
    --model-path outputs/models/adaptive_ppo_agent.pt \
    --timesteps "${PPO_TIMESTEPS:-4000}" \
    --seed "${PPO_SEED:-42}" \
    --horizon "${PPO_HORIZON:-8}"
fi

python -m src.data.coverage \
  --input data/processed/test.csv \
  --output artifacts/unbiased_eval_set.csv \
  --summary-output artifacts/unbiased_summary.json \
  --target-size "${COVERAGE_TARGET_SIZE:-100}" \
  --seed "${FULL_TEST_SEED:-42}"

python -m src.pipeline.run_experiment --mode all "${@}"
