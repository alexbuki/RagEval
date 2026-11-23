"""Three complementary diagnostics for conflicting evidence: ablation, attention, answer likelihood.

Attention is descriptive, not evidence of a model's causal decision mechanism.
Ablations intervene on INPUT documents, not on internal model activations.
"""
import argparse
import csv
import json
import math
from pathlib import Path

CONDITIONS = ("A", "B", "AB", "BA")


def make_prompt(tokenizer, question, source_a, source_b, condition):
    sources = {"A": source_a, "B": source_b}
    blocks = [f"[{label}_START]\n{sources[label]}\n[{label}_END]" for label in condition]
    content = ("Answer the question using the supplied evidence. If the sources disagree, "
               "acknowledge the disagreement and explain briefly.\n\n" +
               "\n\n".join(blocks) + f"\n\nQuestion: {question}")
    messages = [{"role": "system", "content": "Be accurate about the supplied evidence."},
                {"role": "user", "content": content}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    spans = {}
    for label in condition:
        start_marker = f"[{label}_START]\n"
        end_marker = f"\n[{label}_END]"
        start = rendered.index(start_marker) + len(start_marker)
        spans[label] = (start, rendered.index(end_marker, start))
    return rendered, spans


def find_source_tokens(offsets, span):
    start, end = span
    return [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]


def layer_attention(attentions, positions):
    """Attention from final prompt position averaged across heads; measured before generation."""
    rows = []
    for layer, att in enumerate(attentions):
        weights = att[0, :, -1, :].float()
        masses = {label: weights[:, ids].sum(dim=-1).mean().item()
                  for label, ids in positions.items()}
        total = sum(masses.values())
        result = {"layer": layer}
        for label in ("A", "B"):
            mass = masses.get(label)
            result[f"{label.lower()}_mass"] = mass
            result[f"{label.lower()}_mass_per_token"] = mass / len(positions[label]) if mass is not None else None
            result[f"{label.lower()}_share"] = mass / total if mass is not None and total else None
            result[f"{label.lower()}_tokens"] = len(positions.get(label, []))
        rows.append(result)
    return rows


def score_candidate(model, prompt_ids, candidate_ids):
    """Teacher-forced log likelihood of candidate continuation after the same prompt."""
    import torch
    if not candidate_ids:
        raise ValueError("Empty candidate answer")
    continuation = torch.tensor([candidate_ids], dtype=prompt_ids.dtype, device=prompt_ids.device)
    combined = torch.cat((prompt_ids, continuation), dim=1)
    with torch.inference_mode():
        logits = model(input_ids=combined, attention_mask=torch.ones_like(combined), use_cache=False).logits
        # p(candidate_0 | prompt) is at logits[:, prompt_len - 1].
        prompt_len = prompt_ids.shape[1]
        pred_logits = logits[:, prompt_len - 1: -1].float()
        target_ids = continuation
        token_logp = torch.log_softmax(pred_logits, dim=-1).gather(
            -1, target_ids.unsqueeze(-1)
        ).squeeze(-1)
        total = token_logp.sum().item()
    return {"sum_logprob": total, "avg_token_logprob": total / len(candidate_ids),
            "candidate_tokens": len(candidate_ids)}


def relative_preference(score_a, score_b):
    """Two-way normalization of LENGTH-AVERAGED log scores, not probabilities of all answers."""
    delta = score_a["avg_token_logprob"] - score_b["avg_token_logprob"]
    p_a = 1 / (1 + math.exp(-max(-700, min(700, delta))))
    return {"avg_logprob_margin_A_minus_B": delta, "normalized_pair_score_A": p_a,
            "normalized_pair_score_B": 1 - p_a}


def load_cases(args):
    if not args.confrag:
        return [{"id": "demo", "question": "Who is the CEO of ExampleCo now?",
                 "source_a": "In 2023, Alice was CEO of ExampleCo.",
                 "source_b": "In 2025, Bob became CEO of ExampleCo.",
                 "answer_a": "Alice", "answer_b": "Bob"}]
    cases = []
    with open(args.confrag, encoding="utf-8") as f:
        for line in f:
            if len(cases) >= args.num_cases:
                break
            row = json.loads(line)
            answers = row.get("answers", [])
            if len(answers) < 2:
                continue
            websites = {w.get("index"): w.get("content", "") for w in row.get("websites", [])}
            chosen = []
            for answer in answers:
                text = next((websites.get(i, "") for i in answer.get("index", [])
                             if websites.get(i, "").strip()), "")
                if text and answer.get("answer"):
                    chosen.append((answer["answer"], text[:args.source_chars]))
                if len(chosen) == 2:
                    break
            if len(chosen) == 2:
                cases.append({"id": row["id"], "question": row["question"],
                              "source_a": chosen[0][1], "source_b": chosen[1][1],
                              "answer_a": chosen[0][0], "answer_b": chosen[1][0]})
    if not cases:
        raise ValueError("No ConfRAG cases with two annotated answers and source texts")
    return cases


def run_case(model, tokenizer, device, item, args):
    import torch
    rows, attention_rows = [], []
    candidate_ids = {
        "A": tokenizer(" " + item["answer_a"], add_special_tokens=False)["input_ids"],
        "B": tokenizer(" " + item["answer_b"], add_special_tokens=False)["input_ids"],
    }
    for condition in CONDITIONS:
        prompt, spans = make_prompt(tokenizer, item["question"], item["source_a"],
                                    item["source_b"], condition)
        enc = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True,
                        add_special_tokens=False)
        offsets = enc.pop("offset_mapping")[0].tolist()
        input_ids = enc["input_ids"].to(device)
        if input_ids.shape[1] > args.max_input_tokens:
            raise ValueError(f"Case {item['id']}/{condition}: {input_ids.shape[1]} prompt tokens "
                             f"exceed limit {args.max_input_tokens}. Lower --source-chars.")
        positions = {label: find_source_tokens(offsets, span) for label, span in spans.items()}
        if any(not ids for ids in positions.values()):
            raise ValueError(f"Token offsets failed for case {item['id']} condition {condition}")
        with torch.inference_mode():
            forward = model(input_ids=input_ids, attention_mask=enc["attention_mask"].to(device),
                            output_attentions=True, use_cache=False)
            if forward.attentions is None:
                raise RuntimeError("Model did not expose attention weights")
            for a in layer_attention(forward.attentions, positions):
                attention_rows.append({"id": item["id"], "condition": condition, **a})
            del forward
            # Only run attention probe once; free large maps before generation.
            output = model.generate(input_ids=input_ids, attention_mask=enc["attention_mask"].to(device),
                                    do_sample=False,
                                    max_new_tokens=args.max_new_tokens,
                                    pad_token_id=tokenizer.eos_token_id)
            generated = tokenizer.decode(output[0, input_ids.shape[1]:],
                                         skip_special_tokens=True).strip()
            del output
        score_a = score_candidate(model, input_ids, candidate_ids["A"])
        score_b = score_candidate(model, input_ids, candidate_ids["B"])
        preference = relative_preference(score_a, score_b)
        rows.append({"id": item["id"], "condition": condition,
                     "question": item["question"], "answer_a": item["answer_a"],
                     "answer_b": item["answer_b"], "generated_answer": generated,
                     "input_tokens": int(input_ids.shape[1]),
                     "candidate_A_sum_logprob": score_a["sum_logprob"],
                     "candidate_B_sum_logprob": score_b["sum_logprob"],
                     "candidate_A_avg_logprob": score_a["avg_token_logprob"],
                     "candidate_B_avg_logprob": score_b["avg_token_logprob"],
                     "candidate_A_tokens": score_a["candidate_tokens"],
                     "candidate_B_tokens": score_b["candidate_tokens"], **preference})
        print(f"Case {item['id']} {condition}: A score={preference['normalized_pair_score_A']:.3f} "
              f"| {generated[:100]!r}", flush=True)
    return rows, attention_rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--confrag", help="Path to ConfRAG.jsonl; omit to run the small toy example")
    p.add_argument("--num-cases", type=int, default=3)
    p.add_argument("--source-chars", type=int, default=450)
    p.add_argument("--max-input-tokens", type=int, default=768)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--output-dir", default="results")
    args = p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading {args.model} on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required for source token offsets")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="eager", dtype="auto"
    ).to(device).eval()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args)
    # Inputs saved so selection and evidence truncation can be audited.
    (output_dir / "cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    all_scores, all_attentions = [], []
    for item in cases:
        scores, attention = run_case(model, tokenizer, device, item, args)
        all_scores.extend(scores)
        all_attentions.extend(attention)
        write_csv(output_dir / "scores.csv", all_scores)
        write_csv(output_dir / "attention.csv", all_attentions)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"Saved {len(all_scores)} conditions in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
