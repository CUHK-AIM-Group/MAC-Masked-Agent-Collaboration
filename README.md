# MAC Reproducibility (Simple)

This script runs local multi-model collaboration with Ollama:

- script: `Reproducibility-MAC-Masked-Agent-Collaboration.py`
- output: `R_output/NEJMQA_Our_test_v1.out` and `R_output/NEJMQA_Our_test_v1_detailed_model_results.json`

## 1) Install environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Install local Ollama models

Install models with `ollama run xxx` (first run will download model files), for example:

```bash
ollama run qwq:32b
```

Current script uses these models, so install all of them:

```bash
ollama run openthinker:32b
ollama run deepseek-r1:32b
ollama run qwq:32b
ollama run qwen2.5:32b
ollama run phi4:14b
ollama run qwen2.5:14b
```

## 3) Start Ollama service

```bash
ollama serve
```

## 4) Run script

```bash
python Reproducibility-MAC-Masked-Agent-Collaboration.py
```

## 5) Data file required

The script currently reads:

```text
./NEJMQA/NEJMQA-655multi_questions.json
```

Please make sure this file exists before running.

## 6) Quick troubleshooting

- `Connection refused` or model request failed: check `ollama serve` is running.
- `model not found`: run `ollama run <model_name>` first.
- `CUDA out of memory`: reduce number/size of models, or use larger GPU memory.
