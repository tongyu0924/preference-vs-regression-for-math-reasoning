import argparse
import gc
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer

from config import CFG, QUICK_OVERRIDES
from evaluation import best_of_n, make_plots, paired_bootstrap, step_level_metrics, summary_text
from generation import build_trees, free_vllm, make_pairs, sample_test_solutions
from reward_model import Encoder, score_texts, train_reward_model
from utils import PromptBuilder, bf16_ok, load_gsm8k, log, set_seed


def generate_or_load(cfg, pb, train_q, test_q):
    keys = ["policy_model", "n_train_questions", "n_test_questions", "n_cand", "n_roll",
            "max_depth", "temperature", "max_step_tokens", "max_solution_tokens",
            "n_eval_samples", "seed"]
    meta = {k: cfg[k] for k in keys}
    f_tree = os.path.join(cfg["out_dir"], "tree_data.jsonl")
    f_test = os.path.join(cfg["out_dir"], "test_samples.jsonl")
    f_meta = os.path.join(cfg["out_dir"], "generation_meta.json")

    if all(os.path.exists(f) for f in [f_tree, f_test, f_meta]) and json.load(open(f_meta)) == meta:
        log("Using cached generations.")
        return [json.loads(l) for l in open(f_tree)], [json.loads(l) for l in open(f_test)]

    from vllm import LLM
    llm = LLM(model=cfg["policy_model"], dtype="bfloat16" if bf16_ok() else "float16",
              seed=cfg["seed"], gpu_memory_utilization=cfg["vllm_gpu_mem"],
              max_model_len=2048, enforce_eager=True)
    log("Building Q-value trees ...")
    records = build_trees(llm, pb, train_q, cfg)
    log("Sampling test solutions ...")
    test_data = sample_test_solutions(llm, pb, test_q, cfg)
    free_vllm(llm)

    with open(f_tree, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    with open(f_test, "w") as f:
        f.writelines(json.dumps(d) + "\n" for d in test_data)
    json.dump(meta, open(f_meta, "w"))
    return records, test_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small smoke-test run")
    args, _ = ap.parse_known_args()
    cfg = dict(CFG, **(QUICK_OVERRIDES if args.quick else {}))
    os.makedirs(cfg["out_dir"], exist_ok=True)
    set_seed(cfg["seed"])
    assert torch.cuda.is_available(), "A GPU is required."
    device = torch.device("cuda")
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    t0 = time.time()

    train_q, test_q = load_gsm8k(cfg["n_train_questions"], cfg["n_test_questions"], cfg["seed"])
    tok = AutoTokenizer.from_pretrained(cfg["policy_model"])
    pb = PromptBuilder(tok)
    records, test_data = generate_or_load(cfg, pb, train_q, test_q)

    # split by question
    qids = sorted({r["qid"] for r in records})
    random.Random(cfg["seed"]).shuffle(qids)
    heldout = set(qids[:max(1, int(len(qids) * cfg["heldout_frac"]))])
    tr_records = [r for r in records if r["qid"] not in heldout]
    ho_records = [r for r in records if r["qid"] in heldout]
    tr_pairs, ho_pairs = make_pairs(tr_records), make_pairs(ho_records)

    enc = Encoder(tok, pb, cfg["rm_max_len"], device)
    txt = lambda r: enc.text(r["question"], r["prefix"] + [r["step"]])
    train_sets = {"PPM": [(txt(p), txt(n)) for p, n in tr_pairs],
                  "PQM": [(txt(r), r["q"]) for r in tr_records]}

    groups = defaultdict(list)
    for r in records:
        groups[(r["qid"], r["depth"])].append(r["q"])
    nonterm = [r["q"] for r in records if not r["terminal"]]
    data_stats = dict(
        n_records=len(records), n_train_records=len(tr_records), n_heldout_records=len(ho_records),
        n_pairs_train=len(tr_pairs), n_pairs_heldout=len(ho_pairs),
        frac_mixed_nodes=float(np.mean([max(g) > 0 > min(g) for g in groups.values()])),
        frac_q_extreme=float(np.mean([abs(q) == 1 for q in nonterm])) if nonterm else float("nan"),
        test_sample_accuracy=float(np.mean([s["correct"] for d in test_data for s in d["solutions"]])),
    )
    log(f"Data: {json.dumps(data_stats)}")

    # texts to score: held-out candidate steps and every prefix of every test solution
    key = lambda r: (r["qid"], r["depth"], r["step"])
    ho_texts = [txt(r) for r in ho_records]
    flat = [(qi, si, k) for qi, d in enumerate(test_data)
            for si, s in enumerate(d["solutions"]) for k in range(len(s["steps"]))]
    flat_texts = [enc.text(test_data[qi]["question"], test_data[qi]["solutions"][si]["steps"][:k + 1])
                  for qi, si, k in flat]

    step_metrics, final_loss = {}, {}
    for kind in ["PPM", "PQM"]:
        set_seed(cfg["seed"])
        log(f"Training {kind} on {len(train_sets[kind])} examples ...")
        model, losses = train_reward_model(kind, train_sets[kind], enc, cfg, device)
        final_loss[kind] = float(np.mean(losses[-20:]))
        ho_scores = score_texts(model, enc, ho_texts)
        step_metrics[kind] = step_level_metrics(
            {key(r): s for r, s in zip(ho_records, ho_scores)}, ho_records, ho_pairs)
        per_sol = defaultdict(list)
        for (qi, si, _), sc in zip(flat, score_texts(model, enc, flat_texts)):
            per_sol[(qi, si)].append(sc)
        for (qi, si), scs in per_sol.items():
            sol = test_data[qi]["solutions"][si]
            sol[f"{kind} (last step)"] = scs[-1]
            sol[f"{kind} (min step)"] = min(scs)
        log(f"  {kind}: {step_metrics[kind]}")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    ns = [n for n in [1, 2, 4, 8, 16, 32, 64] if n <= cfg["n_eval_samples"]]
    score_keys = ["PPM (last step)", "PQM (last step)", "PPM (min step)", "PQM (min step)"]
    curves, per_q = best_of_n(test_data, score_keys, ns, n_draws=20,
                              rng=np.random.default_rng(cfg["seed"]))
    res = dict(config=cfg, data=data_stats, step_metrics=step_metrics,
               best_of_n=dict(N=ns, curves=curves), final_train_loss=final_loss,
               bootstrap_ppm_minus_pqm=paired_bootstrap(per_q["PPM (last step)"],
                                                        per_q["PQM (last step)"], seed=cfg["seed"]),
               runtime_min=(time.time() - t0) / 60)

    json.dump(res, open(os.path.join(cfg["out_dir"], "results.json"), "w"), indent=2)
    make_plots(ns, curves, step_metrics, cfg["out_dir"])
    summary = summary_text(res)
    print(summary)
    open(os.path.join(cfg["out_dir"], "summary.txt"), "w").write(summary)
    log(f"Done in {res['runtime_min']:.1f} min -> {cfg['out_dir']}/")


if __name__ == "__main__":
    main()
