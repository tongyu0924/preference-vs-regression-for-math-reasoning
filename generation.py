import gc
import random
from collections import defaultdict

from utils import extract_pred, is_correct, log, split_steps


def build_trees(llm, pb, questions, cfg):
    """Expand a step-by-step search tree per question and label every candidate
    step with a Monte Carlo Q-value in [-1, 1]."""
    from vllm import SamplingParams

    step_sp = SamplingParams(n=cfg["n_cand"], temperature=cfg["temperature"], top_p=0.95,
                             max_tokens=cfg["max_step_tokens"], stop=["\n\n"])
    roll_sp = SamplingParams(n=cfg["n_roll"], temperature=cfg["temperature"], top_p=0.95,
                             max_tokens=cfg["max_solution_tokens"])

    states = [dict(qid=i, question=q["question"], gold=q["gold"], prefix=[], done=False)
              for i, q in enumerate(questions)]
    records = []

    for depth in range(cfg["max_depth"]):
        active = [s for s in states if not s["done"]]
        if not active:
            break
        log(f"  depth {depth + 1}: expanding {len(active)} nodes")

        # candidate next steps sharing the same prefix
        outs = llm.generate([pb.with_steps(s["question"], s["prefix"]) for s in active],
                            step_sp, use_tqdm=False)
        cand_per_state = []
        for s, out in zip(active, outs):
            seen, cands = set(), []
            for o in out.outputs:
                step = o.text.strip()
                if not step or step in seen:
                    continue
                seen.add(step)
                hit_eos = o.finish_reason == "stop" and o.stop_reason is None
                terminal = "\\boxed" in step or hit_eos or depth == cfg["max_depth"] - 1
                cands.append(dict(step=step, terminal=terminal,
                                  full="\n\n".join(s["prefix"] + [step])))
            cand_per_state.append(cands)

        # Q-values: exact reward for terminal steps, rollouts otherwise
        roll_prompts, roll_index = [], []
        for si, (s, cands) in enumerate(zip(active, cand_per_state)):
            for ci, c in enumerate(cands):
                if c["terminal"]:
                    c["n_correct"], c["n_total"] = int(is_correct(c["full"], s["gold"])), 1
                else:
                    roll_prompts.append(pb.with_steps(s["question"], s["prefix"] + [c["step"]]))
                    roll_index.append((si, ci))
        if roll_prompts:
            routs = llm.generate(roll_prompts, roll_sp, use_tqdm=False)
            for (si, ci), out in zip(roll_index, routs):
                c = cand_per_state[si][ci]
                c["n_correct"] = sum(is_correct(c["full"] + "\n\n" + o.text, active[si]["gold"])
                                     for o in out.outputs)
                c["n_total"] = len(out.outputs)

        # record all candidates, continue from the best one
        for s, cands in zip(active, cand_per_state):
            if not cands:
                s["done"] = True
                continue
            for c in cands:
                c["q"] = 2.0 * c["n_correct"] / c["n_total"] - 1.0
                records.append(dict(qid=s["qid"], question=s["question"], gold=s["gold"],
                                    depth=depth, prefix=list(s["prefix"]), step=c["step"],
                                    terminal=c["terminal"], q=c["q"],
                                    n_correct=c["n_correct"], n_total=c["n_total"]))
            best = max(cands, key=lambda c: (c["q"], random.random()))
            s["prefix"].append(best["step"])
            if best["terminal"]:
                s["done"] = True
    return records


def make_pairs(records):
    """Up to 2 positives (Q > 0) x 2 negatives (Q < 0) per node, sharing the same prefix."""
    groups = defaultdict(list)
    for r in records:
        groups[(r["qid"], r["depth"])].append(r)
    pairs = []
    for g in groups.values():
        g = sorted(g, key=lambda r: r["q"], reverse=True)
        pos = [r for r in g if r["q"] > 0][:2]
        neg = [r for r in reversed(g) if r["q"] < 0][:2]
        pairs += [(p, n) for p in pos for n in neg]
    return pairs


def sample_test_solutions(llm, pb, questions, cfg):
    from vllm import SamplingParams
    sp = SamplingParams(n=cfg["n_eval_samples"], temperature=cfg["temperature"], top_p=0.95,
                        max_tokens=cfg["max_solution_tokens"])
    outs = llm.generate([pb.head(q["question"]) for q in questions], sp, use_tqdm=False)
    data = []
    for qi, (q, out) in enumerate(zip(questions, outs)):
        sols = []
        for o in out.outputs:
            steps = split_steps(o.text)
            pred = extract_pred(o.text)
            sols.append(dict(text=o.text, steps=steps or [o.text.strip()],
                             correct=bool(is_correct(o.text, q["gold"])),
                             pred=None if pred is None else round(pred, 6)))
        data.append(dict(qid=qi, question=q["question"], gold=q["gold"], solutions=sols))
    return data


def free_vllm(llm):
    import torch
    try:
        from vllm.distributed.parallel_state import (destroy_distributed_environment,
                                                     destroy_model_parallel)
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass
    del llm
    gc.collect()
    torch.cuda.empty_cache()
