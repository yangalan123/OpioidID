import json
from collections import Counter
# expert
with open("expert_data/eval.jsonl", "r", encoding='utf-8') as f_in:
    first_batch_counter = Counter()
    second_batch_counter = Counter()
    for line in f_in:
        obj = json.loads(line)
        if "FirstBatch" in obj['ID']:
            first_batch_counter[obj['answer']] += 1
        else:
            assert "SecondBatch" in obj["ID"]
            second_batch_counter[obj['answer']] += 1
    keys = list(first_batch_counter.keys())
    print("First Batch")
    print(", ".join([f"{k} : {first_batch_counter[k]}" for k in keys]))
    print("Second Batch")
    print(", ".join([f"{k} : {second_batch_counter[k]}" for k in keys]))

# novice
for fn in ["train", "eval"]:
    counter = Counter()
    with open(f"worker_data/{fn}.jsonl", "r", encoding='utf-8') as f_in:
        for line in f_in:
            obj = json.loads(line)
            counter[obj['label'].lower()] += 1
    print(counter)



