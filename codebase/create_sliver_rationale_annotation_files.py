import json
# sliver rationale by T5 model
with open("worker_predictions.json", "r", encoding='utf-8') as f_in:
    sliver_rationale = json.load(f_in)
    id2item = dict()
    for obj in sliver_rationale:
        # arr = obj['translation']['en1'].split("\n\n")
        # assert len(arr) == 3
        _input = obj['translation']['en1']
        _input2 = obj['translation']['en2']
        assert _input.count("\n\nTitle : ") == 1 and _input.count("\n\nText : ") == 1
        assert _input.index("Title : ") < _input.index("Text : ")
        assert _input2.count("Label : ") == 1 and _input2.count("\n\nExplanation : ") == 1
        assert _input2.index("Label : ") < _input2.index("\n\nExplanation : ")
        obj['title'] = _input.split("Title : ")[-1].split("\n\nText : ")[0]
        # assert obj['title'].startswith("Title : ") and obj['title'].count("Title : ") == 1
        # obj['title'] = arr[1].split("Title : ")[-1].strip()
        obj['text'] = _input.split("\n\nText : ")[-1]
        # assert obj['text'].startswith("Text : ") and obj['text'].count("Text : ") == 1
        # obj['text'] = arr[2].split("Text : ")[-1].strip()
        key = f"[title]{obj['title']}[text]{obj['text']}"
        obj['label'] = _input2.split("Label : ")[-1].split("\n\nExplanation : ")[0]
        assert key not in id2item
        id2item[key] = obj

count = 0
with open("eval.jsonl", "r", encoding='utf-8') as f_in:
    with open("eval_sliver_rationale.jsonl", "w", encoding='utf-8') as f_out:
        for line in f_in:
            obj = json.loads(line)
            # assert obj["ID"] in id2item
            key = f"[title]{obj['title']}[text]{obj['text']}"
            if key not in id2item:
                print(f"cannot find ID - {key} in sliver rationale data")
                continue
            predicted_item = id2item[key]
            for key in ['title', "label",  "text"]:
                assert predicted_item[key] == obj[key]
            count += 1
            obj['final_rationale'] = predicted_item["predicted_explanation"]
            f_out.write(json.dumps(obj) + "\n")


assert count == len(id2item)
