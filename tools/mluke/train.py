# Script to fine-tune mLUKE
# saves the trained model to tools/mluke/models/{languages}

# Based on the scripts by huggingface found at:
# https://github.com/huggingface/transformers/tree/main/examples/research_projects/luke

# Original License Text reads as follows:
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import argparse
import math
from pathlib import Path
import json
from timeit import default_timer as timer
from datetime import timedelta

import pandas as pd
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    LukeConfig,
    LukeForEntitySpanClassification,
    MLukeTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)

from datasets import Dataset, concatenate_datasets
from tqdm.auto import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs

import sys

sys.path.insert(0, str(Path.cwd()))
from tools.mluke.utils import (
    stripped_labels,
    entity2id,
    iob_label_list,
    compute_entity_spans_for_luke,
    tokenize_and_pad,
    DataCollatorForLukeTokenClassification,
    evaluate,
)
from utils.registry import load_registry


# Set-up training parameters
def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune (m)LUKE on a span classification task (NER) with the accelerate library"
    )

    parser.add_argument(
        "--languages",
        action="extend",
        nargs="+",
        type=str,
        help="Languages that the model should be trained on. Default: all languages in the registry",
    )

    parser.add_argument(
        "--corpora",
        action="extend",
        nargs="+",
        type=str,
        help="Corpora to use for training the model",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--max_entity_length",
        type=int,
        default=32,
        help=(
            "The maximum total input entity length after tokenization (Used only for (M)Luke models). Sequences longer"
            " than this will be truncated, sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--max_mention_length",
        type=int,
        default=30,
        help=(
            "The maximum total input mention length after tokenization (Used only for (M)Luke models). Sequences"
            " longer than this will be truncated, sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )

    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--label_all_tokens",
        action="store_true",
        help="Setting labels of all special tokens to -100 and thus PyTorch will ignore them.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )

    args = parser.parse_args()

    # Default setting:
    # Include all languages that mLUKE was trained on and are part of this repository
    # (see paper Appendix A for details)
    # https://aclanthology.org/2022.acl-long.505
    if args.languages is None:
        args.languages = ["ar", "de", "en", "es", "fi", "fr", "it", "nl", "pt", "zh"]

    if args.debug:
        print("Running in debug mode!")

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    handler = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[handler])

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load registry and select corpora
    registry = load_registry()
    registry = registry.loc[~(registry.corpus == "wikiann")]

    if args.corpora is None:
        args.corpora = registry.loc[
            registry.language.isin(args.languages)
        ].corpus.unique()

    df_corpora = registry.loc[
        (registry.language.isin(args.languages)) & (registry.corpus.isin(args.corpora))
    ]

    print("Combination:", args.languages, args.corpora)

    # we need to keep track of the sentence ids, because mLUKE
    # splits training samples into smaller chunks and we need to
    # restore the original sentences later
    # Tensorflow can handle simple integers best for sentence ids
    # in this dict, we preserve the original ids
    sentence_id_lookup = {"train": {}, "test": {}, "validation": {}}
    ds = {"train": [], "test": [], "validation": []}
    for _, row in df_corpora.iterrows():
        print(row["path"])
        if row["split"] not in ds.keys():
            continue
        df = pd.read_feather(row["path"])
        df = df.loc[~df.token.isna(), :]
        df = df.loc[df.token.str.len() > 0, :]
        df["new_sentence_id"] = (
            df["corpus"]
            + "_"
            + df["language"]
            + "_"
            + df["sentence_id"]
        )
        sentence_id2int = {
            s_id: i for i, s_id in enumerate(df["new_sentence_id"].unique())
        }
        df["doc_id"] = df["new_sentence_id"].map(sentence_id2int)
        sentence_id_lookup[row["split"]].update(
            {i: s_id for s_id, i in sentence_id2int.items()}
        )
        df["labels"] = df["CoNLL_IOB2"].map(
            {l: i for i, l in enumerate(iob_label_list)}
        )
        df = df.groupby(["doc_id"])[["token", "labels"]].agg(list)
        df = df.rename(columns={"token": "tokens"})
        if args.debug:
            df = df.sample(min(len(df), 200), random_state=1)
        df = df.reset_index()
        # compute sentence boundaries
        # our data is typically already at the sentence level
        # for other datasets, it would be recommended to apply
        # sentence splitting beforehand (e.g., via a spaCy model)
        df["sentence_boundaries"] = df.tokens.apply(lambda x: [0, len(x)])
        ds[row["split"]].append(Dataset.from_dict(df.to_dict(orient="list")))

    print("Loading tokenizer...")
    tokenizer = MLukeTokenizer.from_pretrained(
        "studio-ousia/mluke-base",
        use_fast=False,
        task="entity_span_classification",
        max_entity_length=args.max_entity_length,
        max_mention_length=args.max_mention_length,
    )

    for split in ds.keys():
        ds[split] = concatenate_datasets(ds[split])
        # Ensure document ids are integer
        ds[split] = ds[split].map(
            lambda x: {"doc_id": int(x["doc_id"])},
            desc=f"{split}: Casting doc_id to int",
        )

        ds[split] = ds[split].map(
            compute_entity_spans_for_luke,
            batched=True,
            desc=f"{split}: Adding entity spans",
            fn_kwargs={
                "max_num_subwords": args.max_length,
                "max_entity_length": args.max_entity_length,
                "max_mention_length": args.max_mention_length,
                "label_column_name": "labels",
                "text_column_name": "tokens",
                "doc_id_column_name": "doc_id",
                "tokenizer": tokenizer,
                "entity2id": entity2id,
                "iob_label_list": iob_label_list,
            },
            remove_columns=["sentence_boundaries"],
        )

        ds[split] = ds[split].map(
            tokenize_and_pad,
            batched=True,
            fn_kwargs={
                "max_length": args.max_length,
                "tokenizer": tokenizer,
                "doc_id_column_name": "doc_id",
            },
            remove_columns=ds[split].column_names,
            desc=f"{split}: Running tokenizer on dataset",
        )

    eval_dataset = ds["validation"]
    train_dataset = ds["train"]

    print(f"Loading Model ...")

    if not args.output_dir:
        # path for saving fine-tuned model
        model_dir = Path.cwd() / "tools" / "mluke" / "models"
    else:
        model_dir = Path(args.output_dir)

    if not model_dir.exists():
        model_dir.mkdir(parents=True)

    model_path = model_dir / "_".join(args.languages)
    if not model_path.exists():
        model_path.mkdir(parents=True)

    config = LukeConfig.from_pretrained(
        "studio-ousia/mluke-base", num_labels=len(stripped_labels)
    )

    mluke = LukeForEntitySpanClassification.from_pretrained(
        "studio-ousia/mluke-base", config=config
    )

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorForTokenClassification` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorForLukeTokenClassification(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=args.batch_size
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.batch_size,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in mluke.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in mluke.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    mluke.to(device)

    # Prepare everything with our `accelerator`.
    mluke, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        mluke, optimizer, train_dataloader, eval_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )

    max_train_steps = args.epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    total_batch_size = (
        args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )

    start_training = timer()

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {args.epochs}")
    print(f"  Instantaneous batch size per device = {args.batch_size}")
    print(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0

    for epoch in range(args.epochs):
        mluke.train()
        for step, batch in enumerate(train_dataloader):
            _ = batch.pop("original_entity_spans")
            _ = batch.pop("doc_id")
            _ = batch.pop("context_boundaries")
            outputs = mluke(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= max_train_steps:
                break

        start_evaluation = timer()
        eval_metric = evaluate(mluke, eval_dataloader)
        end_evaluation = timer()
        evaluation_time = timedelta(seconds=end_evaluation - start_evaluation)
        accelerator.print(f"epoch {epoch}:", eval_metric)

    end_training = timer()
    training_time = timedelta(seconds=end_training - start_training)

    print("saving model to", model_path)
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(mluke)
    unwrapped_model.save_pretrained(model_path, save_function=accelerator.save)
    if accelerator.is_main_process:
        model_details = {
            "model": "mluke",
            "model_path": str(model_path),
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "languages": args.languages,
            "corpora": args.corpora,
            "training_duration": training_time.total_seconds(),
            "validation_duration": evaluation_time.total_seconds(),
            "results": eval_metric,
            "args": args.__dict__,
        }

        print(model_details)
        with open(model_path / "model_infos.json", "w", encoding="utf-8") as f:
            json.dump(model_details, f, default=str, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
