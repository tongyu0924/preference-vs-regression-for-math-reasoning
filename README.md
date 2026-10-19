# Preference Learning vs. Q-Value Regression for Mathematical Reasoning

This project compares two ways of training a **process reward model**, a model that scores the individual steps of a math solution, from the same automatically labelled data:

| | Training signal | Loss |
|---|---|---|
| **Q-value-based Process Reward Model (PQM)** | each step is labelled with its Q-value | MSE between predicted score and Q |
| **Process Preference Model (PPM)** | pairs of steps that share the same prefix, one better and one worse | pairwise Bradley–Terry ranking loss |

Both models use the same backbone, the same source of Q-labelled data, and the same optimizer-step and sequence budgets. The training objectives and derived examples differ: the Q-value-based Process Reward Model (PQM) uses labelled steps, while the Process Preference Model (PPM) uses eligible preference pairs.

The setup is a small-scale adaptation of the step-level reward modelling in
[rStar-Math (Guan et al., ICML 2025)](https://arxiv.org/abs/2501.04519).

## Pipeline

1. **Step-level data generation.** A policy model (Qwen2.5-0.5B-Instruct) solves GSM8K training problems step by step.
   - At every step it proposes 4 candidate next steps that share the same prefix.
   - Each candidate gets a Q-value in [-1, 1] from 4 Monte Carlo rollouts (+1 for a correct final answer, -1 for a wrong one).
   - Generation continues from the highest-Q candidate.
2. **Reward model training.** Two reward models are trained on the same Q-values: a Qwen2.5-0.5B-Instruct backbone with LoRA and a scalar tanh head.
   - The **Q-value-based Process Reward Model (PQM)** regresses every labelled step onto its Q-value.
   - The **Process Preference Model (PPM)** is trained on preference pairs. In each node, up to 2 steps with Q > 0 are paired with up to 2 steps with Q < 0.
   - Both run 800 optimizer steps with 8 sequences per step.
3. **Evaluation.**
   - *Step level:* on held-out questions, report pairwise ranking accuracy, Spearman correlation with Q, and MSE w.r.t. Q.
   - *Solution level:* report Best-of-N accuracy on GSM8K test problems. Each reward model picks one of N sampled solutions, using either the score of the final step or the minimum step score. Random choice, majority voting and oracle pass@N serve as references.

## Results: Held-out Pairwise Accuracy

Reported setup: 240 training problems, 250 test problems, 8 sampled solutions per test problem, one seed, and one A100 GPU. The values below are copied from the supplied README and have not been independently rerun. Raw result files were not included in the supplied archive.

| Metric | Process Preference Model (PPM) | Q-value-based Process Reward Model (PQM) |
|---|---:|---:|
| Held-out pairwise accuracy ↑ | **0.214 (21.4%)** | 0.089 (8.9%) |

This experiment compares preference-based training in the Process Preference Model (PPM) with direct Q-value regression in the Q-value-based Process Reward Model (PQM). Held-out pairwise accuracy measures how often each model assigns a higher score to the preferred step than to the rejected step.

## Usage

Requires one CUDA GPU. A T4 works; an A100 is much faster.

```bash
pip install -r requirements.txt
python run.py --quick    # smoke test, a few minutes
python run.py            # full run
```

On Google Colab, the preinstalled `torchao` may be too old for recent `peft`. If importing `peft` fails, run `pip uninstall -y torchao` and restart the runtime.

Generated data is cached in `outputs/`. Re-running with the same generation settings skips the vLLM stage and only retrains and evaluates the reward models.

**Outputs** (in `outputs/`):

| File | Content |
|---|---|
| `summary.txt` | the comparison table |
| `results.json` | all metrics and configuration |
| `best_of_n.png` | Best-of-N accuracy vs. N |
| `step_metrics.png` | step-level metrics |
| `tree_data.jsonl` | all labelled candidate steps with Q-values |
| `test_samples.jsonl` | sampled test solutions |

## Project structure

```
config.py         hyperparameters
utils.py          GSM8K loading, answer checking, prompting
generation.py     step-level search, Monte Carlo Q-values, preference pairs (vLLM)
reward_model.py   Process Preference Model / Q-value-based Process Reward Model training and scoring
evaluation.py     step-level metrics, Best-of-N, bootstrap, plots
run.py            end-to-end pipeline
```

## Limitations

- Preference-pair filtering gives the Process Preference Model (PPM) fewer distinct training examples than the Q-value-based Process Reward Model (PQM), despite matched optimizer-step and sequence budgets.

- The models are small: a 0.5B policy and 0.5B reward models.
- Experiments use GSM8K only and a single seed.
- Q-values come from Monte Carlo rollouts rather than a full MCTS.
- With 4 rollouts per step, Q-value labels are very coarse.

## Reference

Guan, X., Zhang, L. L., Liu, Y., Shang, N., Sun, Y., Zhu, Y., Yang, F., & Yang, M. (2025).
*rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking.* ICML 2025.
