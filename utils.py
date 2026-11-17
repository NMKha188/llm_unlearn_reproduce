# Copyright (C) 2023 ByteDance. All Rights Reserved.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset

torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)


def create_pku_dataloader_from_dataset(tokenizer, dataset, fraction=1.0, batch_size=4):
    data_list = []
    for i in range(len(dataset)):
        if random.random() > fraction:
            continue
        
        item = dataset[i]
        prompt = item["prompt"]
        response_list = []

        if not item["is_response_0_safe"]:
            response_list.append(item["response_0"])
        if not item["is_response_1_safe"]:
            response_list.append(item["response_1"])

        for response in response_list:
            text = f"### Question: {prompt}\n ### Answer: {response}"
            tokenized = tokenizer(text, truncation=True, max_length=512, padding="max_length")
            
            test_text = f"### Question: {prompt}\n ### Answer: "
            test_tokenized = tokenizer(test_text, truncation=True)
            start_loc = len(test_tokenized["input_ids"])

            data_list.append({
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "start_locs": start_loc
            })

    class PKUDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return {
                "input_ids": torch.tensor(self.data[idx]["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(self.data[idx]["attention_mask"], dtype=torch.long),
                "start_locs": torch.tensor(self.data[idx]["start_locs"], dtype=torch.long)
            }

    pku_ds = PKUDataset(data_list)

    def custom_collate_fn(batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        start_locs = torch.stack([item["start_locs"] for item in batch])
        
        labels = input_ids.clone()
        labels[labels == 1] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "start_locs": start_locs,
            "labels": labels
        }

    dataloader = torch.utils.data.DataLoader(
        pku_ds, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True
    )

    return dataloader


def create_truthfulqa_dataloader(tokenizer, batch_size=4):
    """
    Create the TruthfulQA dataloader for the normal data.
    """
    df = pd.read_csv("data/TruthfulQA.csv")
    questions, good_answers = df["Question"].values, df["Best Answer"].values

    data = {"input_ids": [], "attention_mask": []}
    for question, good_answer in zip(questions, good_answers):
        text = f"### Question: {question}\n ### Answer: {good_answer}"
        tokenized = tokenizer(text, truncation=True, max_length=512, padding="max_length")
        data["input_ids"].append(tokenized["input_ids"])
        data["attention_mask"].append(tokenized["attention_mask"])

    dataset = Dataset.from_dict(data)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    train_len = int(0.7 * len(dataset))
    val_len = int(0.1 * len(dataset))
    test_len = len(dataset) - train_len - val_len

    train_data, val_data, test_data = torch.utils.data.random_split(
        dataset, [train_len, val_len, test_len]
    )

    def custom_collate_fn(batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        
        labels = input_ids.clone()
        labels[labels == 1] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True
    )

    return train_dataloader, val_dataloader, test_dataloader


def get_truthfulQA_answers_plaintext(tqa_file_path="data/TruthfulQA.csv"):
    ans_names = ["Best Answer", "Correct Answers", "Incorrect Answers"]
    df = pd.read_csv(tqa_file_path)
    all_ans = []
    for ans_name in ans_names:
        answers = df[ans_name].values
        if ans_name == "Best Answer":
            all_ans.extend(answers)
        else:
            for answer in answers:
                ans_list = answer.split(";")
                for ans in ans_list:
                    all_ans.append(ans.strip())
    return all_ans


def compute_kl(pretrained_model, current_model, batch, device):
    normal_outputs = current_model(
        batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device),
    )

    with torch.no_grad():
        pretrained_outputs = pretrained_model(
            batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )

    prob_p = torch.nn.functional.softmax(pretrained_outputs.logits, -1)
    prob_q = torch.nn.functional.softmax(normal_outputs.logits, -1)

    loss = -(prob_p * torch.log(prob_q + 1e-12)).sum(-1).mean()
    return loss


def get_answer_loss(operation, batch, model, device="cuda:0"):
    assert operation in ["ga", "gd"], "Operation must be either GA or GD."
    input_ids, attention_mask, start_locs, labels = (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["start_locs"],
        batch["labels"].to(device),
    )
    outputs = model(input_ids, attention_mask=attention_mask)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    losses = []
    for bid in range(input_ids.shape[0]):
        one_inp, one_st = input_ids[bid], start_locs[bid]

        position_loss = loss_fct(shift_logits[bid], shift_labels[bid])
        if operation == "ga":
            position_loss = -position_loss

        position_weight = torch.zeros_like(one_inp)
        assert len(position_weight) == len(position_loss) + 1
        position_weight[one_st:] = 1

        position_weight[one_inp == 1] = 0
        if position_weight.sum() > 0:
            position_weight = position_weight / position_weight.sum()

        one_loss = (position_weight[:-1] * position_loss).sum()
        losses.append(one_loss)
    final_loss = torch.stack(losses).mean()
    return final_loss


def get_rand_ans_loss(bad_batch, tokenizer, normal_ans, model, K=5, device="cuda:0"):
    bad_input_ids = bad_batch["input_ids"].to(device)
    rand_ans_list = random.sample(normal_ans, k=K)
    batch_random_features = []
    for batch_idx in range(bad_input_ids.shape[0]):
        single_input_id = bad_input_ids[batch_idx, :]
        ori_text = tokenizer.decode(single_input_id)
        question = ori_text.split("###")[1].split("Question:")[-1].strip()
        question_prefix = f"### Question: {question}\n ### Answer: "
        tokenized_question_prefix = tokenizer(
            question_prefix, truncation=True, max_length=512, padding="max_length"
        )
        start_loc = len(tokenized_question_prefix["input_ids"])

        for rand_ans in rand_ans_list:
            random_sample = f"{question_prefix}{rand_ans}"
            tokenized_rs = tokenizer(
                random_sample, truncation=True, max_length=512, padding="max_length"
            )
            batch_random_features.append(
                {
                    "input_ids": tokenized_rs["input_ids"],
                    "attention_mask": tokenized_rs["attention_mask"],
                    "start_locs": start_loc,
                }
            )

    input_ids = torch.tensor([f["input_ids"] for f in batch_random_features], dtype=torch.long)
    attention_mask = torch.tensor([f["attention_mask"] for f in batch_random_features], dtype=torch.long)
    start_locs = torch.tensor([f["start_locs"] for f in batch_random_features], dtype=torch.long)
    labels = input_ids.clone()
    labels[labels == 1] = -100

    batch_random = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "start_locs": start_locs,
        "labels": labels
    }

    random_loss = get_answer_loss("gd", batch_random, model, device=device)
    return random_loss