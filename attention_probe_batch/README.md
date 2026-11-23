# Conflicting-evidence experiment (three measurements)

**Measures:** (1) attention across transformer layers to source A and B, (2) source ablation and passage-order reversal across `A`, `B`, `AB`, `BA`, (3) model candidate-answer teacher-forced log-likelihood. No CNN training or LightRAG server required.

**Setup** (Python 3.11 and a Conda environment):

```bash
python -m pip install 'torch>=2.3' 'transformers>=4.46,<5' 'accelerate>=1.0' sentencepiece
python -m unittest test_experiments.py
```

**First run:**

```bash
python run_experiments.py --num-cases 1 --output-dir results/demo
```

**With ConfRAG:** from the `attention_probe_batch` directory, assuming the existing project has `data/ConfRAG.jsonl`:

```bash
python run_experiments.py --confrag ../data/ConfRAG.jsonl --num-cases 3 \
  --source-chars 350 --max-input-tokens 768 --output-dir results/confrag
```

You can use `--model Qwen/Qwen2.5-7B-Instruct` on Hive if sufficient memory is available. The default `Qwen/Qwen2.5-0.5B-Instruct` is intentionally smaller for local prototyping. For MPS out-of-memory errors, reduce `--source-chars` to 200, `--max-new-tokens` to 32, or use fewer cases. On CPU, inference can be slow. The first run downloads weights.

**Output:** `cases.json` (exact sources used), `scores.csv` (four conditions per case, generated answers, per-answer log probabilities, and two-choice normalized scores), `attention.csv` (layer-by-layer last-prompt-token attention to A/B), `config.json`.

**Interpretation:** attention mass is the average across heads of attention paid by the *last prompt token* to source text *before answer generation*, not all generation steps. Source length affects mass, so report `mass_per_token` alongside `share`. Single-source conditions have missing A/B attention (blank CSV cells). Candidate answer scores use leading-space continuations and teacher forcing; per-token mean log-likelihood is used to compare answers of different lengths. `normalized_pair_score_A` is normalized **only between two provided candidate strings**, not a calibrated probability of the answer being true. The chosen ConfRAG passages are the first nonempty supporting source for each of the first two answer groups, truncated by character count; they may lose relevant context. Review `cases.json` for suitability. Attention patterns and probability scores do not establish causal internal decision points. Source ablation gives causal evidence of *input sensitivity* only.

**Recommended analysis:** compare `AB` vs `BA` to detect order effects; compare `A` and `B` to show whether the model uses each source individually. Plot `a_share` by layer for AB and BA, and compare the pair-score margins. Run on multiple cases, include non-conflicting controls, and repeat with the same model settings for comparisons.
