import os.path
from collections import Counter
import json
import csv

import torch
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, accuracy_score
from matplotlib import pyplot as plt
font_size = 50
def load_worker_full_annotation():
    with open("worker_data/out_analysis.csv", "r", encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        ret_dict = dict()
        for line in reader:
            key = get_key(line)
            assert key not in ret_dict
            ret_dict[key] = [x.strip() for x in line["labels_aggr"].lower().split(",")]
    return ret_dict

def load_expert_full_annotation():
    with open("ExpertFirstBatch.csv", "r", encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        ret_dict = dict()
        for line in reader:
            key = get_key(line)
            assert key not in ret_dict
            ret_dict[key] = [x.strip() for x in line["label_aggr"].lower().split(",")]
    return ret_dict

def load_content2ID():
    with open("expert_data/eval.jsonl", "r", encoding='utf-8') as f_in:
        content2ID = dict()
        for line in f_in:
            obj = json.loads(line)
            key = get_key(obj)
            content2ID[key] = obj["ID"]
    return content2ID



def get_key(line):
    key = f"[title]{line['title']}[text]{line['text']}"
    return key


def process_t5_files(filepath, full_annotation_dict=None, max_labels_num=2):
    counter = Counter()
    buf = []
    gt = []
    conf_mat_answers = []
    conf_mat_predictions = []
    y_preds = []
    y_trues = []
    label2id = dict()
    global expertContent2ID
    with open(filepath, "r", encoding='utf-8') as reader:
        for line in reader:
            obj = json.loads(line)
            predicted_label = obj["predicted"].strip().lower()
            answer = obj['answer'].strip().lower()
            if predicted_label not in label2id:
                label2id[predicted_label] = len(label2id)
            if answer not in label2id:
                label2id[answer] = len(label2id)
            y_preds.append(predicted_label)
            y_trues.append(answer)
            if answer == predicted_label:
                gt.append({
                    "title": obj["title"],
                    'text': obj['text'],
                    "label": answer,
                })
                if "ID" in obj:
                    gt[-1]["ID"] = obj['ID']
                    assert obj['ID'] == expertContent2ID[get_key(obj)]
                else:
                    if get_key(obj) in expertContent2ID:
                        gt[-1]["ID"] = expertContent2ID[get_key(obj)]
            else:
                err_type = f"{answer} -> {predicted_label}"
                counter[err_type] += 1
                buf.append({
                    "title": obj["title"],
                    'text': obj['text'],
                    "label": answer,
                    "predict": predicted_label,
                })
                if "final_rationale" in obj:
                    buf[-1]["human_explanation"] = obj['final_rationale']
                if "ID" in obj:
                    buf[-1]["ID"] = obj['ID']
                    assert obj['ID'] == expertContent2ID[get_key(obj)]
                else:
                    if get_key(obj) in expertContent2ID:
                        buf[-1]["ID"] = expertContent2ID[get_key(obj)]

            conf_mat_answers.append(answer)
            conf_mat_predictions.append(predicted_label)


    plt.rcParams.update({"font.size": font_size})
    disp = ConfusionMatrixDisplay.from_predictions(conf_mat_answers, conf_mat_predictions, xticks_rotation="vertical", normalize="true")
    fig = disp.ax_.get_figure()
    fig.set_figwidth(20)
    fig.set_figheight(16)
    plt.title("t5-11b\n" + os.path.basename(filepath).split(".json")[0])
    plt.tight_layout()
    plt.savefig(f"{filepath.replace('.json', '_')}ConfusionMatrix_prop_larger_font_size.pdf")
    plt.show()
    plt.clf()
    disp = ConfusionMatrixDisplay.from_predictions(conf_mat_answers, conf_mat_predictions, xticks_rotation="vertical")
    fig = disp.ax_.get_figure()
    fig.set_figwidth(20)
    fig.set_figheight(16)
    plt.tight_layout()
    plt.savefig(f"{filepath.replace('.json', '_')}ConfusionMatrix_larger_font_size.pdf")
    plt.clf()
    print("acc: ", len(gt) / (len(gt) + len(buf)))
    print("macro f1: ", f1_score(y_trues, y_preds, average="macro"))
    print("classwise f1: ", f1_score(y_trues, y_preds, average=None))
    # get classwise accuracy, not compute tp,fp,fn,tn
    classwise_acc = dict()
    classwise_num = dict()
    ytrues_category = {"unanimous": [], "disagreement": [], "arguable(all)": []}
    ypreds_category = {"unanimous": [], "disagreement": [], "arguable(all)": []}
    for label in label2id:
        classwise_acc[label] = 0
        classwise_num[label] = 0
    for y_i in range(len(y_trues)):
        if y_trues[y_i] == y_preds[y_i]:
            classwise_acc[y_trues[y_i]] += 1
        classwise_num[y_trues[y_i]] += 1
    for label in label2id:
        classwise_acc[label] /= classwise_num[label]
    for label in sorted(list(label2id.keys())):
        print(f"classwise acc for {label}: {classwise_acc[label]}")

    print("label2id:", label2id)
    if full_annotation_dict is not None:
        counter_all = Counter()
        counter_correct = Counter()
        for line in gt:
            key = get_key(line)
            if key in full_annotation_dict:
                label_set = set(full_annotation_dict[key])
                assert len(label_set) <= max_labels_num
                if len(label_set) == 1:
                    counter_all["unanimous"] += 1
                    counter_correct["unanimous"] += 1
                    ytrues_category["unanimous"].append(line["label"])
                    ypreds_category["unanimous"].append(line["label"])
                else:
                    counter_all["disagreement"] += 1
                    counter_correct["disagreement"] += 1
                    ytrues_category["disagreement"].append(line["label"])
                    ypreds_category["disagreement"].append(line["label"])
                    ypreds_category["arguable(all)"].append(line["label"])
                    ytrues_category["arguable(all)"].append(line["label"])
        fixed_count = 0
        for line in buf:
            key = get_key(line)
            if key in full_annotation_dict:
                label_set = set(full_annotation_dict[key])
                assert len(label_set) <= max_labels_num
                if len(label_set) == 1:
                    counter_all["unanimous"] += 1
                    ytrues_category["unanimous"].append(line["label"])
                    ypreds_category["unanimous"].append(line["predict"])
                else:
                    counter_all["disagreement"] += 1
                    ytrues_category["disagreement"].append(line["label"])
                    ypreds_category["disagreement"].append(line["predict"])
                    ytrues_category["arguable(all)"].append(line["label"])
                    ypreds_category["arguable(all)"].append(line["predict"])
                    if line["predict"].lower() in label_set:
                        fixed_count += 1
                        ytrues_category['arguable(all)'][-1] = line["predict"].lower()
        for _type in ["unanimous", "disagreement"]:
            print(f"{_type} acc: {counter_correct[_type] / counter_all[_type]} ({counter_correct[_type]} / {counter_all[_type]})")
        print("arguable(all) acc:", (fixed_count + counter_correct['disagreement']) / counter_all['disagreement'])
        for _type in ["unanimous", "disagreement"]:
            print(f"{_type} macro f1: {f1_score(ytrues_category[_type], ypreds_category[_type], average='macro')} ")
        print("arguable(all) macro f1:", {f1_score(ytrues_category['arguable(all)'], ypreds_category['arguable(all)'], average='macro')})
        print(counter_all)

    return counter, buf, gt

def process_deberta_files(filepath, worker_type, setting, full_annotation_dict=None, check_data_in_full=True, max_labels_num=2):
    source_data_dict = {
        "expert": "expert_data",
        "worker": "worker_data"
    }
    source_fn = os.path.join(source_data_dict[worker_type], f"eval_{setting}.json")
    with open(source_fn, "r", encoding='utf-8') as f_in:
        buf = []
        for line in f_in:
            buf.append(json.loads(line))
        source_data = buf
    data = torch.load(filepath)
    assert len(data) == len(source_data)
    counter = Counter()
    buf = []
    gt = []
    conf_mat_answers = []
    conf_mat_predictions = []
    y_preds = []
    y_trues = []
    global expertContent2ID
    for i, predict_data in enumerate(data):
        text = predict_data['text']
        assert text.count("[text]") == 1 and text.count("[title]") == 1 and text.count("[rationale]") <= 1
        assert text == source_data[i]["text"]
        answer = source_data[i]["label"].strip().lower()
        predicted_label = predict_data["predict"].strip().lower()
        title = text.split("[text]")[0].split("[title]")[1]
        post = text.split("[text]")[1].split("[rationale]")[0]
        y_preds.append(predicted_label)
        y_trues.append(answer)
        if answer == predicted_label:
            gt.append({
                "title": title,
                'text': post,
                "label": answer,
            })
            item_key = get_key(gt[-1])
            if item_key in expertContent2ID:
                gt[-1]["ID"] = expertContent2ID[item_key]
        else:
            err_type = f"{answer} -> {predicted_label}"
            counter[err_type] += 1
            buf.append({
                "title": title,
                'text': post,
                "label": answer,
                "predict": predicted_label
            })
            item_key = get_key(buf[-1])
            if item_key in expertContent2ID:
                buf[-1]["ID"] = expertContent2ID[item_key]
        conf_mat_answers.append(answer)
        conf_mat_predictions.append(predicted_label)

    plt.rcParams.update({"font.size": font_size})
    disp = ConfusionMatrixDisplay.from_predictions(conf_mat_answers, conf_mat_predictions, xticks_rotation="vertical", normalize="true")
    fig = disp.ax_.get_figure()
    fig.set_figwidth(20)
    fig.set_figheight(16)
    plt.title(f"DeBERTa\n {worker_type} {setting}")
    plt.tight_layout()
    plt.savefig(f"{setting}_ConfusionMatrix_prop_larger_font_size.pdf")
    plt.show()
    plt.clf()
    plt.rcParams.update({"font.size": font_size})
    disp = ConfusionMatrixDisplay.from_predictions(conf_mat_answers, conf_mat_predictions, xticks_rotation="vertical")
    fig = disp.ax_.get_figure()
    fig.set_figwidth(20)
    fig.set_figheight(16)
    plt.tight_layout()
    plt.savefig(f"{setting}_ConfusionMatrix_larger_font_size.pdf")
    plt.clf()
    print("acc: ", len(gt) / (len(gt) + len(buf)))
    print("macro f1: ", f1_score(y_trues, y_preds, average="macro"))
    ytrues_category = {"unanimous": [], "disagreement": [], "arguable(all)": []}
    ypreds_category = {"unanimous": [], "disagreement": [], "arguable(all)": []}
    if full_annotation_dict is not None:
        counter_all = Counter()
        counter_correct = Counter()
        for line in gt:
            key = get_key(line)
            if check_data_in_full:
                assert key in full_annotation_dict
            else:
                if key not in full_annotation_dict:
                    continue
            label_set = set(full_annotation_dict[key])
            assert len(label_set) <= max_labels_num
            if len(label_set) == 1:
                counter_all["unanimous"] += 1
                counter_correct["unanimous"] += 1
                ytrues_category["unanimous"].append(line["label"])
                ypreds_category["unanimous"].append(line["label"])
            else:
                counter_all["disagreement"] += 1
                counter_correct["disagreement"] += 1
                ytrues_category["disagreement"].append(line["label"])
                ypreds_category["disagreement"].append(line["label"])
                ypreds_category["arguable(all)"].append(line["label"])
                ytrues_category["arguable(all)"].append(line["label"])
        fixed_count = 0
        for line in buf:
            key = get_key(line)
            if check_data_in_full:
                assert key in full_annotation_dict
            else:
                if key not in full_annotation_dict:
                    continue
            label_set = set(full_annotation_dict[key])
            assert len(label_set) <= max_labels_num
            if len(label_set) == 1:
                counter_all["unanimous"] += 1
                ytrues_category["unanimous"].append(line["label"])
                ypreds_category["unanimous"].append(line["predict"])
            else:
                counter_all["disagreement"] += 1
                ytrues_category["disagreement"].append(line["label"])
                ypreds_category["disagreement"].append(line["predict"])
                ytrues_category["arguable(all)"].append(line["label"])
                ypreds_category["arguable(all)"].append(line["predict"])
                if line["predict"].lower() in label_set:
                    fixed_count += 1
                    ytrues_category['arguable(all)'][-1] = line["predict"].lower()
        for _type in ["unanimous", "disagreement"]:
            print(f"{_type} acc: {counter_correct[_type] / counter_all[_type]} ({counter_correct[_type]} / {counter_all[_type]}) ")
        print("arguable(all) acc:", (fixed_count + counter_correct['disagreement']) / counter_all['disagreement'])

        for _type in ["unanimous", "disagreement"]:
            print(f"{_type} macro f1: {f1_score(ytrues_category[_type], ypreds_category[_type], average='macro')} ")
        print("arguable(all) macro f1:", {f1_score(ytrues_category['arguable(all)'], ypreds_category['arguable(all)'], average='macro')})

    return counter, buf, gt

def compute_batch_wise_accuracy(buf, gt, full_annotation_dict, batch_i="SecondBatch", check_data_in_full=False, max_labels_num=2):
    # only for few-shot data
    if full_annotation_dict is not None:
        counter_all = Counter()
        counter_correct = Counter()
        for line in gt:
            key = get_key(line)
            assert "ID" in line
            if batch_i not in line["ID"]:
                continue

            if check_data_in_full:
                assert key in full_annotation_dict
            if key in full_annotation_dict:
                # first batch
                label_set = set(full_annotation_dict[key])
            else:
                # second batch
                label_set = {line["label"], }
            assert len(label_set) <= max_labels_num
            if len(label_set) == 1:
                counter_all["unanimous"] += 1
                counter_correct["unanimous"] += 1
            else:
                counter_all["disagreement"] += 1
                counter_correct["disagreement"] += 1
        fixed_count = 0
        for line in buf:
            key = get_key(line)
            assert "ID" in line
            if batch_i not in line["ID"]:
                continue
            if check_data_in_full:
                assert key in full_annotation_dict
            if key in full_annotation_dict:
                # first batch
                label_set = set(full_annotation_dict[key])
            else:
                # second batch
                label_set = {line["label"], }
            assert len(label_set) <= max_labels_num
            if len(label_set) == 1:
                counter_all["unanimous"] += 1
            else:
                counter_all["disagreement"] += 1
                if line["predict"].lower() in label_set:
                    fixed_count += 1
        for _type in ["unanimous", "disagreement"]:
            if counter_all[_type] == 0:
                continue
            print(f"{batch_i}\n {_type} acc: {counter_correct[_type] / counter_all[_type]} ({counter_correct[_type]} / {counter_all[_type]}) ")
        if counter_all["disagreement"] > 0:
            print("arguable(all) acc:", (fixed_count + counter_correct['disagreement']) / counter_all['disagreement'])

if __name__ == '__main__':
    # process t5 files
    common_err_types = set()
    common_err_types_rank_inv = dict()
    combined_err_posts = dict()
    key2content = dict()
    all_model_types = set()
    worker_annotation_full_dict = load_worker_full_annotation()
    expert_annotation_full_dict = load_expert_full_annotation()
    expertContent2ID = load_content2ID()
    for worker_type in ["worker", "expert"]:
        for setting in ["with_explanations", "without_explanations"]:
            t5_filepath = os.path.join("exp_results", "T5-11B", f"{worker_type}_pred_{setting}.json")
            model_type = f"predict_{worker_type}_{setting}"
            print(f"processing {t5_filepath}")
            if worker_type == "expert":
                err_counter, error_buf, gt = process_t5_files(t5_filepath, full_annotation_dict=expert_annotation_full_dict, max_labels_num=3)
                compute_batch_wise_accuracy(error_buf, gt, expert_annotation_full_dict, batch_i='SecondBatch', check_data_in_full=False, max_labels_num=3)
            else:
                err_counter, error_buf, gt = process_t5_files(t5_filepath, full_annotation_dict=worker_annotation_full_dict, max_labels_num=2)
            print(err_counter.most_common())
            for item in error_buf:
                key = f"[title]{item['title']}[text]{item['text']}"
                if key not in combined_err_posts:
                    combined_err_posts[key] = []
                    assert key not in key2content
                    key2content[key] = {
                        "title": item['title'],
                        'text': item['text'],
                        "label": item['label']
                    }
                if len(combined_err_posts[key]) > 0:
                    assert key2content[key]['label'] == item['label']
                combined_err_posts[key].append({
                    "model_type": model_type,
                    "predict": item["predict"]
                })
                all_model_types.add(model_type)
            for i, item in enumerate(err_counter.most_common()):
                if item[0] not in common_err_types_rank_inv:
                    common_err_types_rank_inv[item[0]] = 0
                common_err_types_rank_inv[item[0]] += 1 / (i + 1)
            if len(common_err_types) == 0:
                common_err_types = set([x[0] for x in err_counter.most_common(10)])
            else:
                common_err_types &= set([x[0] for x in err_counter.most_common(10)])
            with open(t5_filepath.replace(".json", "_err2.csv"), "w", encoding='utf-8', newline='') as writer:
                fieldnames = ["title", "text", "label", model_type]
                if "human_explanation" in error_buf[0]:
                    fieldnames = ["title", "text", "label", "human_explanation", model_type]
                dict_writer = csv.DictWriter(writer, fieldnames=fieldnames)
                dict_writer.writeheader()
                for line in error_buf:
                    new_obj = line.copy()
                    new_obj[model_type] = line['predict']
                    new_obj.pop("predict")
                    if "ID" in new_obj:
                        new_obj.pop("ID")
                    dict_writer.writerow(new_obj)
    print(common_err_types)
    common_err_types_rank_inv_list = [(x, common_err_types_rank_inv[x] / 4) for x in common_err_types]
    common_err_types_rank_inv_list.sort(key=lambda x: x[1], reverse=True)
    print(common_err_types_rank_inv_list)
    t5_filepath = os.path.join("exp_results", "T5-11B")
    all_model_types = list(all_model_types)
    with open(os.path.join(t5_filepath, "combined_error_analysis.csv"), "w", encoding='utf-8', newline='') as f_out:
        fieldnames = ['title', 'text', 'label']
        for error_type in all_model_types:
            fieldnames.append("err" + error_type.strip("predict"))
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for key in combined_err_posts:
            content = key2content[key]
            new_obj = content.copy()
            err_model_types = {x["model_type"]: x["predict"] for x in combined_err_posts[key]}
            for _err_model_type in all_model_types:
                if _err_model_type in err_model_types:
                    new_obj["err" + _err_model_type.strip("predict")] = err_model_types[_err_model_type]
                else:
                    new_obj["err" + _err_model_type.strip("predict")] = "N/A"
            writer.writerow(new_obj)

    # process deberta files
    common_err_types = set()
    common_err_types_rank_inv = dict()
    combined_err_posts = dict()
    key2content = dict()
    all_model_types = set()
    for worker_type in ["worker", "expert"]:
        for setting in ["w_explanation", "wo_explanation"]:
            deberta_filepath = os.path.join(f"worker_data/deberta-v3-large/{setting}/lr_2e-5_epoch_10/{worker_type}_eval_{setting}", "predict_results_None.pkl")
            model_type = f"predict_{worker_type}_{setting}"
            print(f"processing {deberta_filepath}")
            if worker_type == 'expert':
                err_counter, error_buf, gt = process_deberta_files(deberta_filepath, worker_type, setting, full_annotation_dict=expert_annotation_full_dict, max_labels_num=3, check_data_in_full=False)
                compute_batch_wise_accuracy(error_buf, gt, expert_annotation_full_dict, batch_i='SecondBatch', check_data_in_full=False, max_labels_num=3)
            else:
                err_counter, error_buf, gt = process_deberta_files(deberta_filepath, worker_type, setting, full_annotation_dict=worker_annotation_full_dict)
            print(err_counter.most_common())
            for item in error_buf:
                key = f"[title]{item['title']}[text]{item['text']}"
                if key not in combined_err_posts:
                    combined_err_posts[key] = []
                    assert key not in key2content
                    key2content[key] = {
                        "title": item['title'],
                        'text': item['text'],
                        "label": item['label']
                    }
                if len(combined_err_posts[key]) > 0:
                    assert key2content[key]['label'] == item['label']
                combined_err_posts[key].append({
                    "model_type": model_type,
                    "predict": item["predict"]
                })
                all_model_types.add(model_type)
            for i, item in enumerate(err_counter.most_common()):
                if item[0] not in common_err_types_rank_inv:
                    common_err_types_rank_inv[item[0]] = 0
                common_err_types_rank_inv[item[0]] += 1 / (i + 1)
            if len(common_err_types) == 0:
                common_err_types = set([x[0] for x in err_counter.most_common(10)])
            else:
                common_err_types &= set([x[0] for x in err_counter.most_common(10)])
            with open(deberta_filepath.replace("_None.pkl", "_err.csv"), "w", encoding='utf-8', newline='') as writer:
                dict_writer = csv.DictWriter(writer, fieldnames=["title", "text", "label", model_type])
                dict_writer.writeheader()
                for line in error_buf:
                    new_obj = line.copy()
                    new_obj[model_type] = line['predict']
                    new_obj.pop("predict")
                    if "ID" in new_obj:
                        new_obj.pop("ID")
                    dict_writer.writerow(new_obj)
    print(common_err_types)
    common_err_types_rank_inv_list = [(x, common_err_types_rank_inv[x] / 4) for x in common_err_types]
    common_err_types_rank_inv_list.sort(key=lambda x: x[1], reverse=True)
    print(common_err_types_rank_inv_list)
    deberta_filepath = f"LargeBatch/finetuning_data/deberta-v3-large"
    all_model_types = list(all_model_types)
    with open(os.path.join(deberta_filepath, "combined_error_analysis.csv"), "w", encoding='utf-8', newline='') as f_out:
        fieldnames = ['title', 'text', 'label']
        for error_type in all_model_types:
            fieldnames.append("err" + error_type.strip("predict"))
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for key in combined_err_posts:
            content = key2content[key]
            new_obj = content.copy()
            err_model_types = {x["model_type"]: x["predict"] for x in combined_err_posts[key]}
            for _err_model_type in all_model_types:
                if _err_model_type in err_model_types:
                    new_obj["err" + _err_model_type.strip("predict")] = err_model_types[_err_model_type]
                else:
                    new_obj["err" + _err_model_type.strip("predict")] = "N/A"
            writer.writerow(new_obj)




