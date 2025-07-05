
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import datasets
from utils import get_model_identifiers_from_yaml, add_dataset_index

def convert_raw_data_to_model_format(tokenizer, max_length,  question, answer, model_configs):
    question_start_token, question_end_token, answer_token = model_configs['question_start_tag'], model_configs['question_end_tag'], model_configs['answer_tag']
    new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

    #change label to -100 for question tokens
    for i in range(num_question_tokens): label[i] = -100

    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)
    


class TextForgetDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family,  max_length=512, split = "forget10", loss_type="idk"):
        super(TextForgetDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.forget_data = datasets.load_dataset(data_path, split)["train"]
        retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        self.retain_data =datasets.load_dataset(data_path, retain_split)["train"]
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type

        if self.loss_type == "idk":
            self.split1, self.split2 = "idk", "retain"
            self.idontknowfile = "tofu/data/idontknow.jsonl"
            self.idk = open(self.idontknowfile, "r").readlines()
        else:
            self.split1, self.split2 = "forget", "retain"

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in [self.split1, self.split2]:
            #use questions from forget set if split is idk or forget
            data = self.retain_data if data_type == "retain" else self.forget_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            question = data[idx]['question']
            answer = data[idx]['answer']

            if data_type == "idk":
                #get a random answer position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()
                
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            rets.append(converted_data)
        return rets


class TextForgetDatasetDPOQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = "forget10", ):
        super(TextForgetDatasetDPOQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.forget_data = datasets.load_dataset(data_path, split)["train"]
        self.idontknowfile = "data/idontknow.jsonl"
        self.idk = open(self.idontknowfile, "r").readlines()
        retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        self.retain_data = datasets.load_dataset(data_path, retain_split)["train"]
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []

        for data_type in ["idk", "forget", "retain"]:
            data = self.forget_data if data_type != "retain" else self.retain_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            
            question = data[idx]['question']
            
            if data_type != "idk":
                answer = data[idx]['answer']
            else:
                #get a random position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()

            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            rets.append(converted_data)
        return rets


class TextDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = None, question_key='question', answer_key='answer'):
        super(TextDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # data_len = len(datasets.load_dataset(data_path, split)["train"])
        # self.data = datasets.load_dataset(data_path, split)["train"].select(range(min(100, data_len)))
        self.data = datasets.load_dataset(data_path, split)["train"]

        self.data = add_dataset_index(self.data)
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]
        indices = self.data[idx]['index']
        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])


        return torch.stack(pad_input_ids_list).squeeze(),\
                torch.stack(label_list).squeeze(),\
                torch.stack(pad_attention_mask_list).squeeze(),\
                torch.tensor(indices)






from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
import random

class WMDPUnlearnDataset(Dataset):
    def __init__(self, tokenizer):
        super(WMDPUnlearnDataset, self).__init__()
        self.tokenizer = tokenizer

        # Load and preprocess the datasets
        self.retain_data = self.load_and_preprocess("retain")
        self.forget_data = self.load_and_preprocess("forget")

    def load_and_preprocess(self, dataset_type):
        # Define dataset paths based on type
        dataset_paths = {
            "retain": [
                ("cais/wmdp-corpora", "cyber-retain-corpus"),
                ("cais/wmdp-corpora", "bio-retain-corpus")
            ],
            "forget": [
                ("cais/wmdp-corpora", "cyber-forget-corpus"),
                ("json", "data/wmdp/wmdp-corpora/bio_remove_dataset.jsonl")
            ]
        }

        # Load and concatenate datasets
        datasets = []
        for path, dataset_name in dataset_paths[dataset_type]:
            if path == "json":
                loaded_dataset = load_dataset("json", data_files=dataset_name, split="train", cache_dir="./.cache")
            else:
                loaded_dataset = load_dataset(path, dataset_name, cache_dir="./.cache")["train"]
            datasets.append(loaded_dataset)

        concatenated = concatenate_datasets(datasets)

        # Preprocess and set the dataset format
        processed_dataset = concatenated.map(self.preprocess, batched=True, remove_columns=concatenated.column_names)
        processed_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return processed_dataset

    def preprocess(self, examples):
        # Apply tokenization and potentially other transformations
        tokenized = self.tokenizer(
            examples['text'],
            max_length=1024,
            truncation=True,
            padding='max_length',
            add_special_tokens=True
        )
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': tokenized['input_ids']
        }

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        forget_sample = self.forget_data[idx]
        retain_sample = self.retain_data[random.randint(0, len(self.retain_data) - 1)]  # random sampling for retain data
        return forget_sample, retain_sample



class WMDPUnlearnDatasetWiki(Dataset):
    def __init__(self, tokenizer):
        super(WMDPUnlearnDatasetWiki, self).__init__()
        self.tokenizer = tokenizer

        # Load and preprocess the datasets
        self.retain_data = self.load_and_preprocess("retain")
        self.forget_data = self.load_and_preprocess("forget")

    def load_and_preprocess(self, dataset_type):
        # Define dataset paths based on type
        dataset_paths = {
            "retain": [
            # Replace previous retain corpora with WikiText
            ("wikitext", "wikitext-2-raw-v1")
        ],
            "forget": [
                ("cais/wmdp-corpora", "cyber-forget-corpus"),
                ("json", "data/wmdp-corpora/bio_remove_dataset.jsonl")
            ]
        }

        # Load and concatenate datasets
        datasets = []
        for path, dataset_name in dataset_paths[dataset_type]:
            if path == "json":
                loaded_dataset = load_dataset("json", data_files=dataset_name, split="train", cache_dir="./.cache")
            elif path == "wikitext":
                loaded_dataset = load_dataset(
                "wikitext",
                dataset_name,
                split="test", 
                cache_dir="./.cache"
            )
            else:
                loaded_dataset = load_dataset(path, dataset_name, cache_dir="./.cache")["train"]
            datasets.append(loaded_dataset)

        concatenated = concatenate_datasets(datasets)
        concatenated = concatenated.filter(lambda x: len(x["text"]) > 50)

        # Preprocess and set the dataset format
        processed_dataset = concatenated.map(self.preprocess, batched=True, remove_columns=concatenated.column_names)
        processed_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return processed_dataset

    def preprocess(self, examples):
        # Apply tokenization and potentially other transformations
        tokenized = self.tokenizer(
            examples['text'],
            max_length=2048, #1024
            truncation=True,
            padding='max_length',
            add_special_tokens=True
        )
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': tokenized['input_ids']
        }

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        forget_sample = self.forget_data[idx]
        retain_sample = self.retain_data[random.randint(0, len(self.retain_data) - 1)]  # random sampling for retain data
        return forget_sample, retain_sample


def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return input_ids, attention_masks


def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)

def custom_data_collator_with_indices(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    indices = [s[3] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask), torch.stack(indices)



def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss










import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from pathlib import Path
import datasets
import torch.nn.functional as F

# Assuming the necessary helper functions such as `add_dataset_index` and `convert_raw_data_to_model_format` are imported

class TextDatasetQARDF(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split=None, 
                 question_key='question', answer_key='answer', splits=1, selected_split=1, mode=1,dataset_split = "forget05"):
        """
        Parameters:
        - data_path: path to the dataset
        - tokenizer: tokenizer to process text
        - model_family: model family for config
        - max_length: maximum length of input sequences
        - split: which split of the data to use (train, validation, etc.)
        - question_key: key for question data in the dataset
        - answer_key: key for answer data in the dataset
        - splits: the total number of splits to create from the data
        - selected_split: the specific split to use (e.g., 1 for the first split, 2 for the second)
        - mode: 1 for including the selected split, 2 for excluding the selected split
        """
        super(TextDatasetQARDF, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_configs = get_model_identifiers_from_yaml(model_family)

        self.data = datasets.load_dataset(data_path, dataset_split)["train"]

        self.data = add_dataset_index(self.data)
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key
        
        # Calculate split size
        total_len = len(self.data)
        split_size = total_len // splits

        # Define the range of the selected split
        start_idx = split_size * (selected_split - 1)
        end_idx = start_idx + split_size if selected_split < splits else total_len

        # Select the split(s) based on mode
        if mode == 1:
            # Mode 1: Include the selected split
            self.data = self.data.select(range(start_idx, end_idx))
            print(len(self.data))
        elif mode == 0:
            # Mode 2: Exclude the selected split, include the rest
            exclude_indices = list(range(start_idx, end_idx))
            self.data = self.data.filter(lambda x: x['index'] not in exclude_indices)
            print(len(self.data))
        # else:
        #     raise ValueError("Invalid mode. Use 1 for including and 2 for excluding the selected split.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]
        indices = self.data[idx]['index']
        
        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])

        return torch.stack(pad_input_ids_list).squeeze(),\
               torch.stack(label_list).squeeze(),\
               torch.stack(pad_attention_mask_list).squeeze(),\
               torch.tensor(indices)

    def get_collate_fn(self):
        def collate_fn(batch):
            input_ids, labels, attention_masks, indices = zip(*batch)
            return {
                'input_ids': torch.stack(input_ids),
                'labels': torch.stack(labels),
                'attention_masks': torch.stack(attention_masks),
                'indices': torch.stack(indices)
            }
        return collate_fn










