# Architecture

The framework combines:
1. Data preparation and transparent label engineering.
2. Rule/policy safety controls.
3. XGBoost scoring module.
4. Ollama SLM (`gemma4:e2b`) adaptive controller.
5. Explanation and optional content generation modules.
6. Experiment runner with ablation modes.
7. FastAPI serving layer.
