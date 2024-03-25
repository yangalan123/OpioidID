from functools import partial
from typing import *
import cleantext
import numpy as np

def rationale_filtering(rationales: List[str]):
    length = len(rationales)
    if length <= 1:
        return rationales
    banned_list = set()
    for i in range(length):
        for j in range(i+1, length):
            if len(rationales[i]) != len(rationales[j]):
                if rationales[i].strip() in rationales[j].strip():
                    banned_list.add(i)
                    break
                if rationales[j].strip() in rationales[i].strip():
                    banned_list.add(j)
            else:
                if rationales[i].strip() == rationales[j].strip():
                    banned_list.add(i)
                    continue
    buf = []
    for i in range(length):
        if i not in banned_list:
            buf.append(rationales[i])
    return buf


def default_prompt_formatting(sample: dict, add_answer=False, add_rationale=True):
    base_str = f"Title: {sample['title']}\nText: {sample['text']}\n"
    rationale = None
    if "rationale" in sample:
        rationale = "\n".join(set([x.strip() for x in rationale_filtering(sample["rationale"])]))
        if len(rationale) == 0:
            rationale = "None"

    if not add_answer:
        return base_str
    else:
        if rationale is not None and add_rationale:
            return base_str + f"Opioid Usage Label: {sample['answer']}\n" + f"Explanation: {rationale}\n\n"
        else:
            return base_str + f"Opioid Usage Label: {sample['answer']}\n\n"


def prepare_prompts(selected_samples: list[dict], instruction:str, format_func=None):
    ret = instruction
    if format_func is None:
        format_func = partial(default_prompt_formatting, add_answer=True)
    for sample_id, sample in enumerate(selected_samples):
        ret = ret + f"{sample_id + 1}) " + format_func(sample)
    return ret


def normalize_text(s: str) -> str:
    s = cleantext.fix_bad_unicode(s)
    s = cleantext.replace_urls(s, "[url]")
    s = cleantext.remove_emoji(s)
    s = cleantext.replace_emails(s, '[SomeEmail]')
    s = cleantext.replace_phone_numbers(s, "[PhoneNumber]")
    return s


def longest_str_intersection(a: str, b: str):
    # find maximum possible intersection contagious substring
    # divide-and-conquer, expected complexity O(nmlogn)
    # divide-and-conquer + kmp should be able to do O(nlogn), but I am too lazy to write it :-)
    # DP should be able to do O(nm), but I am too lazy to write it :-)
    def check_substring(k: int):
        ret = []
        flag = False
        for i in range(len(a) - k + 1):
            if a[i: i + k] in b:
                ret.append(a[i: i + k])
                flag = True
        return flag, ret

    if len(a) == 0 or len(b) == 0:
        return True, []
    maximum_possible_length = min(len(a), len(b))
    left = 0
    right = maximum_possible_length
    ret = []
    while left <= right:
        mid = (left + right) // 2
        flag, temp = check_substring(mid)
        if flag:
            ret = temp
            left = mid + 1
        else:
            right = mid - 1
    return ret


def fleiss_kappa(M):
    """Computes Fleiss' kappa for group of annotators.
    :param M: a matrix of shape (:attr:'N', :attr:'k') with 'N' = number of subjects and 'k' = the number of categories.
        'M[i, j]' represent the number of raters who assigned the 'i'th subject to the 'j'th category.
    :type: numpy matrix
    :rtype: float
    :return: Fleiss' kappa score
    """
    N, k = M.shape  # N is # of items, k is # of categories
    n_annotators = float(np.sum(M[0, :]))  # # of annotators
    tot_annotations = N * n_annotators  # the total # of annotations
    category_sum = np.sum(M, axis=0)  # the sum of each category over all items

    # chance agreement
    p = category_sum / tot_annotations  # the distribution of each category over all annotations
    PbarE = np.sum(p * p)  # average chance agreement over all categories

    # observed agreement
    P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
    Pbar = np.sum(P) / N  # add all observed agreement chances per item and divide by amount of items

    return round((Pbar - PbarE) / (1 - PbarE), 4)
