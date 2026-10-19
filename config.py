CFG = dict(
    policy_model="Qwen/Qwen2.5-0.5B-Instruct",
    rm_model="Qwen/Qwen2.5-0.5B-Instruct",
    n_train_questions=240,
    n_test_questions=250,
    heldout_frac=0.15,
    n_cand=4,              # candidate next steps per node
    n_roll=4,              # Monte Carlo rollouts per candidate
    max_depth=8,
    temperature=0.8,
    max_step_tokens=200,
    max_solution_tokens=512,
    n_eval_samples=8,      # Best-of-N pool size
    rm_train_steps=800,    # same number of optimizer steps for both reward models
    rm_lr=1e-4,
    rm_head_lr=1e-3,
    rm_batch=8,
    rm_max_len=640,
    lora_r=16,
    lora_alpha=32,
    seed=42,
    out_dir="outputs",
    vllm_gpu_mem=0.55,
)

QUICK_OVERRIDES = dict(n_train_questions=24, n_test_questions=24, n_eval_samples=2,
                       max_depth=5, rm_train_steps=150, out_dir="outputs_quick")

SYSTEM_PROMPT = ("Please reason step by step, separating each step with a blank line, "
                 "and put your final answer within \\boxed{}.")
