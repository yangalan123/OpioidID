import json
import random
from nltk import sent_tokenize
# for source_fn, target_fn in zip(["train.jsonl", "eval.jsonl"], ['train_w_explanation.json', "eval_w_explanation.json"]):
# for source_fn, target_fn in zip(["eval_sliver_rationale.jsonl"], ["eval_w_sliver_explanation.json"]):
seed = 5
random.seed(seed)
for source_fn, target_fn in zip(["eval.jsonl"], [f"eval_w_random_explanation_{seed}.json"]):
# for source_fn, target_fn in zip(["train.jsonl", "eval.jsonl"], ['train_wo_explanation.json', "eval_wo_explanation.json"]):
    with open(source_fn, "r", encoding='utf-8') as f_in:
        with open(target_fn, "w", encoding='utf-8') as f_out:
            for line in f_in:
                obj = json.loads(line)
                new_obj = dict()
                if "w_explanation" in target_fn or "w_sliver_explanation" in target_fn:
                    new_obj["text"] = f"[title]{obj['title']}[text]{obj['text']}[rationale]{obj['final_rationale']}"
                else:
                    if "w_random_explanation" in target_fn:
                        sents = sent_tokenize(obj["title"] + obj['text'])
                        random_rationale = random.sample(sents, 1)[0]
                        new_obj["text"] = f"[title]{obj['title']}[text]{obj['text']}[rationale]{random_rationale}"
                    else:
                        new_obj["text"] = f"[title]{obj['title']}[text]{obj['text']}"
                new_obj["label"] = obj["label"]
                # if using rationale, then uncomment the following line
                # new_obj["text_rationale"] = obj["rationale"]
                f_out.write(json.dumps(new_obj) + "\n")
