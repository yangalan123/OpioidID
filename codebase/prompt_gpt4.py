import os
from functools import partial
import shutil
import sys
from datetime import datetime
import json
import openai
import nltk
from loguru import logger
from utils import default_prompt_formatting, prepare_prompts
import tiktoken

import os
import openai
import json
import requests
import time
requests.adapters.DEFAULT_RETRIES = 5
requests.adapters.HTTPAdapter.pool_connections = 9999
requests.adapters.HTTPAdapter.pool_maxsize = 9999
requests.adapters.HTTPAdapter.max_retries = 5
openai.api_timeout = 1200

def count_tokens(model, workload, **kwargs):
    def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
        """Returns the number of tokens used by a list of messages."""
        # official codes from OpenAI Cookbook: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        if model == "gpt-3.5-turbo":
            # print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
            return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
        elif model == "gpt-4":
            # print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
            return num_tokens_from_messages(messages, model="gpt-4-0314")
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif model == "gpt-4-0314":
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    if "turbo" in model or "gpt-4" in model:
        return num_tokens_from_messages(workload, model)
    else:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(workload))


def prepare_chat_workload(workload, **kwargs):
    if "system" in kwargs:
        system_msg = kwargs['system']
    else:
        system_msg = "You are a helpful assistant."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": workload}
    ]
    return messages




openai.api_key = ""
ic_sample_flag = "manual_without_rationale"
# ic_sample_flag = "manual_with_rationale"

# start_sequence = "\n Drug Usage Label:"
# prompt = "Given the following text please identify the appropriate drug usage category they describe. Your answer should be only among the following categories: 'Medical Use', 'Misuse', 'Recovery', 'Relapse' and 'Addiction'.\n\n1) I've been using heroin since 7th grade and im in 11th grade now And at this point I'm a week clean and I want to stay clean but it's fucking hard as shit. The only thing keeping me off it is a promise I made to my cousin. I guess I just really need support right now. I'm in a rough place and I really hate how things are going for me.\nDrug Usage Label: Recovery\n\n2) Who else takes more than 700mg to feel the good effects of trammies. Honestly i always go above 600-700 and i have never had a seizure on it. I am scared that one day i might actually end up seizuring in a public place.\nDrug Usage Label: Misuse\n\n3) So I’ve been doing 1-2 grams of heroin a day for the last year or two  My mental health is so low I can’t seem to put up with withdrawals as it amplifies my suicidal tendencies.  I just moved and don’t really know anyone where I’m at so I can’t get anything. Is there anything I can do to help with my withdrawals?\nDrug Usage Label: Addiction\n\n4) I wish I could take a picture of my high and post it as high porn. Hnnnnnng. It was worth being clean for 90 days.. it was worth being born and living life until that point... for that high. Wow.And then it was followed by guilt, remorse, shame and a headache today :(\nDrug Usage Label: Relapse\n\n5) I have been taking hydrocodone for about 5-6 years and have never had the problems I have with Oxycodone. I had surgery 2  days ago and it has not been a fun ride with the Oxycodone. Is it normal to get what feels like Jolts through your body at any given time? Also my blood pressure has been pretty low since taking the Oxycodone. I didn’t know if it was the medicine causing my blood pressure to be low or just having surgery recently.But the scariest thing is the Jolts through the body suddenly. Also if something loud on TV or in my room happens it makes me jump. Which stuff like that has never bothered me.  Also a side note I take benzos with my pain medication.Thank you for your time.\nDrug Usage Label: Medical Use\n\n6) 5ish months after getting clean, im manager at my job making good money, have a better relationship w my family, and am getting ready to go to school. but the depression and anxiety are hitting me hard and other shit going on in my life also is not helping. i just hit someone up for narcan incase i decide to relapse. i really just feel like i need to vent but i cant vent to anyone that i know personally so hi r/opiates im back after a few months.\nDrug Usage Label: Recovery\n\n7) When you know you need to stretch your supply, what helps you stay disciplined enough to do so? I used to be able to stretch my supply (hydro and tramadol) so that I could save them for the weekend but as my addiction has increased I feel like I can never not get high whenever I have pills in my possession. I have ankylosing spondylitis (spinal fusion) and I see a PM doctor but only get a 60 count of 5-325 norco a month. I can never make it last 2 weeks and the rest of the month just sucks so bad. I can never find a connect so I'm always SOL and would like to be able to stretch my script over that month. What do you all do?\nDrug Usage Label: Addiction\n\n8) I've read that the combination can kill you, but I've also read that it can turn out to be a pretty good time. I can't find information on what amount of each substance can be fatal when combined. I have 35mg of oxy (haven't taken it in 2 months, so little to no tolerance). I do drink regularly though, so my tolerance for that is pretty high. I plan on taking all of the oxy, so how much alcohol will be too much?\\\" what drug usage category does it describe?\nDrug Usage Label: Misuse\n\n9) I just need to tell someone to get it off my chest.   Been clean for two years. Got clean in rehab in south Florida after being in a six month program. Got a job in a new place with new faces and worked the program for a minute but kinda let it go after a while. So I started taking Kratom and I really liked it obviously. First mistake. Then my friend was gonna have a bachelor party back in my home town and I was like fuck yeah I’ll go back and just do one 30. Why not? So yeah I get back that evening and do one from my old guy and it turns out it’s pressed. Nose starts gushing immediately and I overdose. Now my friend saved my life but I don’t know... just feel so fucking stupid. Two fucking years. What a waste.   Don’t start taking chances.\nDrug Usage Label: Relapse\n\n10) So I went to the e.r yesterday cause I was having terrible pain all over my face I assume it’s my teeth or something. I walked in there expecting nothing at all because the doctor could not figure out what my pain was from. But he looked at my records and saw that I had oxycodone prescribed to me back in 2018 for my gall bladder removal and he just says “since you’ve had it before here you go” and prescribed me 20 5 mg oxys which is at least a little something haha. I don’t understand, I guess I just got lucky. I hope everyone has a good day!\nDrug Usage Label: Medical Use\n\n11) Here I am. 4:30am. I tried to stay asleep until 7:00 and get well before heading uni. I couldn't, i know I have dope next to me. It's calling me. NOW, NOW NOW. So I just couldn't wait any longer and did 40 mg IV morphine with a needle so used, I'm surprised it even punctured my skin. But the rush was great, and maybe in a couple hours, it'll be as good too.Happy nods.\nDrug Usage Label: Addiction\n\n12) I live in queensland and have been on opiates for my fibromyalgia on and off for ten years. last week my doctor said the government requires me to submit to random drug tests. i didnt really feel comfortable asking what they would be testing for. is there anyone who knows the actual info on this?\nDrug Usage Label: Medical Use\n\n"
# instruction without explanation
# instruction = "Given the following text please identify the appropriate drug usage category they describe. Your answer should be only among the following categories: 'Medical Use', 'Misuse', 'Recovery', 'Relapse', 'Addiction', 'None'.\n\n"
# instruction with explanation
# instruction = "Given the following title and text please identify the appropriate drug usage category among the following types: 'Medical Use', 'Misuse', 'Recovery', 'Relapse', 'Addiction', 'None'.  Please provide an explanation for your answer by extracting the relevant span from the text that justifies your choice.\n\n"
if "with_rationale" in ic_sample_flag:
    instruction = "Given the following title and text please identify the appropriate opioid usage category among the following types: 'medical use', 'misuse', 'recovery', 'relapse', 'addiction', 'not using'.  Please provide an explanation for your answer by extracting the relevant span from the text that justifies your choice.\n\n"
else:
    if "without_rationale" in ic_sample_flag:
        instruction = "Given the following title and text please identify the appropriate opioid usage category among the following types: 'medical use', 'misuse', 'recovery', 'relapse', 'addiction', 'not using'.\n\n"
    else:
        print(f"UNKNOWN IC SAMPLE FLAG: {ic_sample_flag}")
        exit()
selected_samples = []
sample_ids = set()
# ic_sample_flag = "manual_with_rationale"
# prepare training data, 13 expert-annotated samples
with open(f"expert_data/train.jsonl", 'r', encoding='utf-8') as f_in:
    for line in f_in:
        obj = json.loads(line)
        selected_samples.append(obj)
        sample_ids.add(obj['ID'])
if "with_rationale" in ic_sample_flag:
    # by default, using rationale
    prompt = prepare_prompts(instruction=instruction, selected_samples=selected_samples)
else:
    prompt = prepare_prompts(instruction=instruction, selected_samples=selected_samples,
                             format_func=partial(default_prompt_formatting, add_rationale=False, add_answer=True))
print(prompt)


count = 0
total = 0
ignore = 0

# novice-annotated data
# filepath = "worker_data/eval.jsonl"
# expert-annotated data
filepath = "expert_data/eval.jsonl"

results = []
# engine = 'text-davinci-002'
# engine = 'text-curie-001'
engine="gpt-4"
log_dir = f"logs/[your_log_dir]/{'worker' if 'LargeBatch' in filepath else 'expert'}_{engine}_{ic_sample_flag}_{datetime.now().strftime('%y_%m')}"

os.makedirs(log_dir, exist_ok=True)
script_name = os.path.basename(__file__)
shutil.copyfile(script_name, os.path.join(log_dir, script_name))
shutil.copyfile("utils.py", os.path.join(log_dir, "utils.py"))

logger.add(os.path.join(log_dir, "log.txt"))
logger.info("Prompt Start")
logger.info(prompt)
logger.info("Prompt Ends")
empty = 0
costs = []
all_data = []
response = None
for line in open(filepath, "r", encoding='utf-8'):
    line = json.loads(line)
    _input = f"{len(selected_samples) + 1}) " + default_prompt_formatting(line, add_answer=False)
    if "LargeBatch" not in filepath:
        if line["ID"] in sample_ids:
            logger.warning("ignore post at: " + line["ID"])
            ignore += 1
            continue
    else:
        line["answer"] = line["label"].lower()
    if "[deleted]" in line['title'] or '[deleted]' in line['text']:
        logger.warning("ignore deleted post at: " + line["ID"])
        empty += 1
        continue
    total += 1
    _id = line["ID"] if "worker" not in log_dir else f"WORKER_{total}"
    logger.info(_input, line["answer"])
    if os.path.exists(os.path.join(log_dir, f"{_id}.json")):
        try:
            logger.info(f"Loading from cache: {_id}.json")
            with open(os.path.join(log_dir, f"{_id}.json"), "r", encoding='utf-8') as f_in:
                obj = json.load(f_in)
            results.append(obj)
            assert obj['answer'] == line['answer'] and obj['ID'] == _id, f"Mismatched ID ({_id} vs {line['ID']}) or answer: " + str(obj) + "\n" + str(line)
            if obj['prediction'] == obj['answer']:
                count += 1
            logger.info(f"Loaded {_id}.json successfully, current accuracy: {count / total}")
            continue
        except Exception as e:
            logger.warning(f"Failed to load {_id}.json, re-running")
    else:
        logger.info(f"Running {_id}.json as it does not exist")

    try:
        workload = prepare_chat_workload(prompt + _input)
        response = openai.ChatCompletion.create(model=engine, messages=workload, temperature=0,
                                                max_tokens=20, top_p=1, n=1,
                                                stop=["."])
        num_tokens = count_tokens(model="gpt-4", workload=workload)
        cost = num_tokens / 1000 * 0.03 + 50 / 1000 * 0.06
        costs.append(cost)
        content = response['choices'][0]['message']['content']
        logger.info("Got Response" + content.split("Opioid Usage Label:")[1].lower())
        prediction = content.split("Opioid Usage Label:")[1].lower()
        if prediction.lstrip().rstrip() == line["answer"]:
            count = count + 1
        results.append({"text": _input, "input": _input, "prediction": prediction.lstrip().rstrip(),
                        "answer": line["answer"], "response": response,
                        "ID": line["ID"] if "worker" not in log_dir else f"WORKER_{total}"})
        json.dump(results[-1], open(os.path.join(log_dir, f"{_id}.json"), "w", encoding='utf-8'), indent=4)
        logger.info("Dump to " + os.path.join(log_dir, f"{_id}.json"))
        logger.info(f"acc: {count / total}, ignore: {ignore}, empty: {empty}, total: {total}")
    except Exception as e1:
        logger.warning(e1)
        logger.warning(response)
        logger.warning("full text input failed, try to trim the input...")
        try:
            text = nltk.sent_tokenize(_input)
            text = ''.join(text[:-int(0.25 * len(text))])
            workload = prepare_chat_workload(prompt + text)
            response = openai.ChatCompletion.create(model=engine, messages=workload, temperature=0,
                                                    max_tokens=20, top_p=1, n=1,
                                                    stop=["."])
            content = response['choices'][0]['message']['content']
            logger.info("Got Response" + content.split("Opioid Usage Label:")[1].lower())
            prediction = content.split("Opioid Usage Label:")[1].lower()
            if prediction.lstrip().rstrip() == line["answer"]:
                count = count + 1
            results.append({"text": _input, "input": text, "prediction": prediction.lstrip().rstrip(),
                            "answer": line["answer"], "response": response,
                            "ID": line["ID"] if "worker" not in log_dir else f"WORKER_{total}"})
            json.dump(results[-1], open(os.path.join(log_dir, f"{_id}.json"), "w", encoding='utf-8'), indent=4)
            logger.info("Dump to " + os.path.join(log_dir, f"{_id}.json"))
            logger.info(f"acc: {count / total}, ignore: {ignore}, empty: {empty}, total: {total}")
        except Exception as e:
            logger.warning(response)
            logger.warning(e)
            logger.warning("Failed")
    time.sleep(1)

with open(os.path.join(log_dir, f"output.json"), "w") as f:
    f.write(json.dumps(results, indent=4) + '\n')

logger.info(f"acc: {count / total}, ignore: {ignore}, empty: {empty}, total: {total}")
