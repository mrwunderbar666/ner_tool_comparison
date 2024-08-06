# Script to evaluate a fine tuned mLUKE model

import argparse
from pathlib import Path
import json
from timeit import default_timer as timer
from datetime import timedelta

import pandas as pd
from torch.utils.data import DataLoader
from transformers import (
    LukeConfig,
    LukeForEntitySpanClassification,
    MLukeTokenizer,
    default_data_collator,
)

from datasets import Dataset
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
        description="Evaluate an (m)LUKE on a span classification task."
    )

    parser.add_argument("model_path", help="Directory where the model is stored.")

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

    # Check for model

    model_path = Path(args.model_path)
    assert model_path.exists()
    model_infos = model_path / "model_infos.json"
    assert model_infos.exists(), f"MODEL INFO DOES NOT EXIST: {model_infos}"
    with open(model_infos) as f:
        infos = json.load(f)

    results_destination = (
        Path.cwd() / "results" / f"mluke_{'_'.join(infos['languages'])}.csv"
    )

    # Load registry and select corpora
    registry = load_registry()
    registry = registry.loc[registry.split == "validation"]

    if args.corpora is None:
        args.corpora = registry.loc[
            registry.language.isin(args.languages)
        ].corpus.unique()

    df_corpora = registry.loc[
        (registry.language.isin(args.languages)) & (registry.corpus.isin(args.corpora))
    ]

    print("Evaluating following languages:", args.languages)
    print("Corpora for evaluation:", args.corpora)

    print("Loading tokenizer...")
    tokenizer = MLukeTokenizer.from_pretrained(
        "studio-ousia/mluke-base",
        use_fast=False,
        task="entity_span_classification",
        max_entity_length=infos["args"]["max_entity_length"],
        max_mention_length=infos["args"]["max_mention_length"],
    )

    print("Loading and processing validation sets ...")

    validation_sets = {}
    for _, row in df_corpora.iterrows():
        print(row["path"])
        df = pd.read_feather(row["path"])
        df = df.loc[~df.token.isna(), :]
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

        ds = Dataset.from_dict(df.to_dict(orient="list"))
        # Ensure document ids are integer
        ds = ds.map(
            lambda x: {"doc_id": int(x["doc_id"])}, desc=f"Casting doc_id to int"
        )

        ds = ds.map(
            compute_entity_spans_for_luke,
            batched=True,
            desc=f"Adding entity spans",
            fn_kwargs={
                "max_num_subwords": infos["args"]["max_length"],
                "max_entity_length": infos["args"]["max_entity_length"],
                "max_mention_length": infos["args"]["max_mention_length"],
                "label_column_name": "labels",
                "text_column_name": "tokens",
                "doc_id_column_name": "doc_id",
                "tokenizer": tokenizer,
                "entity2id": entity2id,
                "iob_label_list": iob_label_list,
            },
            remove_columns=["sentence_boundaries"],
        )

        ds = ds.map(
            tokenize_and_pad,
            batched=True,
            fn_kwargs={
                "max_length": infos["args"]["max_length"],
                "tokenizer": tokenizer,
                "doc_id_column_name": "doc_id",
            },
            remove_columns=ds.column_names,
            desc=f"Running tokenizer on dataset",
        )

        validation_sets[row["path"]] = {
            "dataset": ds,
            "language": row["language"],
            "corpus": row["corpus"],
            "subset": row["subset"],
            "tokens": row["tokens"],
            "sentences": row["sentences"],
        }

    print(f"Loading Model ...")

    config = LukeConfig.from_pretrained(
        args.model_path, num_labels=len(stripped_labels)
    )

    mluke = LukeForEntitySpanClassification.from_pretrained(
        args.model_path, config=config
    )

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    mluke.to(device)

    # DataLoaders creation:
    if infos["args"]["pad_to_max_length"]:
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

    end_results = []
    for path, v in validation_sets.items():
        print("Evaluating", path)
        dataloader = DataLoader(
            v["dataset"],
            collate_fn=data_collator,
            batch_size=infos["args"]["batch_size"],
        )

        # Prepare everything with our `accelerator`.
        mluke, dataloader = accelerator.prepare(mluke, dataloader)

        start_evaluation = timer()
        results = evaluate(mluke, dataloader)
        end_evaluation = timer()
        validation_time = timedelta(seconds=end_evaluation - start_evaluation)

        r = [{"task": key, **val} for key, val in results.items() if type(val) == dict]
        overall = {
            k.replace("overall_", ""): v for k, v in results.items() if type(v) != dict
        }
        overall["task"] = "overall"
        r.append(overall)
        r = pd.DataFrame(r)
        r["validation_corpus"] = path
        r["validation_duration"] = validation_time.total_seconds()
        r["language"] = v["language"]
        r["corpus"] = v["corpus"]
        r["subset"] = v["subset"]
        r["tokens"] = v["tokens"]
        r["sentences"] = v["sentences"]
        r["model_path"] = infos["model_path"]
        r["model_languages"] = ", ".join(infos["languages"])
        r["model_corpora"] = ", ".join(infos["corpora"])
        end_results.append(r)

    eval_results = pd.concat(end_results)
    eval_results.to_csv(results_destination, index=False)

    print(80 * "-", "\n")

    print("Done!")


if __name__ == "__main__":
    main()
