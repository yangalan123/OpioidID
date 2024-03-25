import csv
import os
import json
import matplotlib.pyplot as plt
import glob
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from collections import Counter

def createContext2IDmapping():
    res = {}
    with open("expert_data/eval.jsonl", "r") as f_in:
        for line in f_in:
            obj = json.loads(line)
            assert obj["title"] + obj["text"] not in res
            res[obj['title'] + obj['text']] = obj["ID"]
    return res

root_path = "logs/[YOUR_LOG_DIR]/expert_gpt-4_manual_without_rationale"
if "GPT4" not in root_path:
    with open(f"{root_path}/log.txt", "r", encoding='utf-8') as f_in:
        failed_count = 0
        ignore_count = 0
        for line in f_in:
            if "WARNING" in line and "Failed" in line:
                failed_count += 1
            if "WARNING" in line and "ignore" in line:
                ignore_count += 1
        print("failed", failed_count, "ignore", ignore_count)

if "worker" not in root_path:
    c2id_dict = createContext2IDmapping()
else:
    c2id_dict = None
mismatch_case = {}
f_out = open(os.path.join(root_path, "error_analysis_with_rationale.csv"), "w", encoding='utf-8', newline='')
writer = csv.DictWriter(f_out, fieldnames=["text", "answer", "prediction", "ID"])
writer.writeheader()
conf_mat_predictions = []
conf_mat_answers = []
log_dir = root_path
with open(f"{log_dir}/output.json", "r", encoding='utf-8') as f_in:
    res = json.load(f_in)
    correct_count = 0
    total_count = 0
    ambig_count = 0
    categories = {}
    batch_categories = {'First': {'correct': 0, "total":0}, "Second": {"correct": 0, "total": 0}}

    for item in res:
        original_prediction = item['prediction']
        item["prediction"] = item['prediction'].split("\n")[0]
        if item["prediction"] == "withdrawal":
            item["prediction"] = "recovery"
        if item["prediction"] == "overdose":
            item["prediction"] = "addiction"
        if item["prediction"] == "recreational":
            item["prediction"] = "addiction"
        if item['prediction'] == "1) relapse, 2) relapse, 3":
            item['prediction'] = "relapse"
        if item['prediction'] in ["misusing", "misue"]:
            item['prediction'] = "misuse"
        if item['prediction'] in ["detox", "taper"]:
            continue
        if item["prediction"] != original_prediction:
            print("change in label", item['prediction'], original_prediction)
        if item["answer"] != "ambiguous":
            total_count += 1
            conf_mat_answers.append(item["answer"])
            conf_mat_predictions.append(item["prediction"])
        if item['answer'] not in categories:
            categories[item['answer']] = {"correct": 0, "total": 0}
        if item["prediction"] == item['answer']:
            if item["answer"] != "ambiguous":
                correct_count += 1
            categories[item['answer']]['correct'] += 1
        else:
            if item['answer'] not in mismatch_case:
                mismatch_case[item['answer']] = []
            if item["answer"] != "ambiguous":
                writer.writerow({
                    "text": item["input"],
                    "prediction": item["prediction"],
                    "answer": item["answer"],
                    "ID": item["ID"]
                })
            mismatch_case[item['answer']].append((item['prediction'], ))
        if "ID" in item:
            id_indicator = item["ID"]
        else:
            _title = item["text"].split("Title: ")[1].split("Text: ")[0].strip()
            _text = item['text'].split("Text: ")[1].strip()
            if c2id_dict is not None:
                id_indicator = c2id_dict[_title + _text]
            else:
                id_indicator = "Worker-{}"

        if "First" in id_indicator:
            batch_flag = "First"
        else:
            batch_flag = "Second"
        batch_categories[batch_flag]["total"] += 1
        if item["prediction"] == item['answer']:
            batch_categories[batch_flag]['correct'] += 1

        categories[item['answer']]['total'] += 1
    for category in categories:
        categories[category]['acc'] = categories[category]['correct'] / (categories[category]['total'] + 1e-6)
    for category in batch_categories:
        batch_categories[category]['acc'] = batch_categories[category]['correct'] / (batch_categories[category]['total'] + 1e-6)
    print("decomposition", json.dumps(categories, indent=4))
    print("batch decomposition", json.dumps(batch_categories, indent=4))
    print("correct count", correct_count)
    print("total count", total_count)
for key in mismatch_case:
    mismatch_case[key] = list(Counter((mismatch_case[key])).most_common())
print(json.dumps(mismatch_case, indent=4))

print("fixed accuracy: ", correct_count / total_count)
print("f1 score", f1_score(conf_mat_answers, conf_mat_predictions, average='macro'))
f_out.close()


plt.rcParams.update({"font.size": 30})
disp = ConfusionMatrixDisplay.from_predictions(conf_mat_answers, conf_mat_predictions, xticks_rotation="vertical", normalize="true")
fig = disp.ax_.get_figure()
fig.set_figwidth(20)
fig.set_figheight(16)
plt.tight_layout()
plt.savefig(f"{log_dir}/ConfusionMatrix.pdf")
plt.show()


