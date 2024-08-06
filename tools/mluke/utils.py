# Collections of utilities for fine-tuning and evaluating mLUKE

# Based on the scripts by huggingface found at:
# https://github.com/huggingface/transformers/tree/main/examples/research_projects/luke
# Published under Apache 2.0 License.
# Copyright 2022 The HuggingFace Inc. team.

# And on source code obtained from mLUKE's original authors:
# https://github.com/studio-ousia/luke
# Published under Apache 2.0 License
# Copyright 2022 Studio Ousia.


from typing import List, Tuple, Dict, Union, Literal, Optional, Iterable
import unicodedata
import math
import warnings
import itertools
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import MLukeTokenizer, LukeForEntitySpanClassification
from transformers.models.luke.modeling_luke import EntitySpanClassificationOutput
from transformers.data.data_collator import DataCollatorMixin
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from datasets import ClassLabel
from datasets.formatting.formatting import LazyBatch
import numpy as np
from seqeval.scheme import IOB2, Entities

sys.path.insert(0, str(Path.cwd()))
from utils.metrics import compute_metrics

stripped_labels = ["O", "PER", "ORG", "LOC", "MISC"]
iob_label_list = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
]

id2entity = {i: e for i, e in enumerate(stripped_labels)}
entity2id = {v: k for k, v in id2entity.items()}


def is_punctuation(char: str) -> bool:
    cp = ord(char)
    if (
        (cp >= 33 and cp <= 47)
        or (cp >= 58 and cp <= 64)
        or (cp >= 91 and cp <= 96)
        or (cp >= 123 and cp <= 126)
        or cp == 180
    ):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def padding_tensor(
    sequences: List[int],
    padding_value: Union[int, tuple],
    padding_side: Literal["right", "left"],
    sequence_length: int,
) -> List[int]:
    """
    Apply padding to a tensor

    sequences: input tensor to be padded
    padding_value: value used for padding (e.g., -1 for adding -1)
    padding_side: pad left or right on sequences
    sequence_length: total length of output sequence
    """

    if isinstance(padding_value, tuple):
        out_tensor = np.full((len(sequences), sequence_length, 2), padding_value)
    else:
        out_tensor = np.full((len(sequences), sequence_length), padding_value)

    for i, tensor in enumerate(sequences):
        if padding_side == "right":
            if isinstance(padding_value, tuple):
                out_tensor[i, : len(tensor[:sequence_length]), :2] = tensor[
                    :sequence_length
                ]
            else:
                out_tensor[i, : len(tensor[:sequence_length])] = tensor[
                    :sequence_length
                ]
        else:
            if isinstance(padding_value, tuple):
                out_tensor[i, len(tensor[:sequence_length]) - 1 :, :2] = tensor[
                    :sequence_length
                ]
            else:
                out_tensor[i, len(tensor[:sequence_length]) - 1 :] = tensor[
                    :sequence_length
                ]

    return out_tensor.tolist()


def compute_sentence_boundaries_for_luke(
    examples: LazyBatch, text_column_name="tokens"
) -> LazyBatch:
    """
    This might not work with every dataset, since not all were split at the
    sentence level

    Output:
        - all input columns
        - `sentence_boundaries` (List[int, int]): For each unit a list of two integers (beginning and end positions)
    """

    sentence_boundaries = []

    for tokens in examples[text_column_name]:
        sentence_boundaries.append([0, len(tokens)])

    examples["sentence_boundaries"] = sentence_boundaries

    return examples


# TODO: Update docstring, correct IO keys!
# TODO: Account for context_boundaries. If input sequence is very long
#           use europeana as example case
def compute_entity_spans_for_luke(
    examples: LazyBatch,
    max_num_subwords=510,
    max_entity_length=32,  # maximum length for entity_spans, this is the tensor that contains all possible entity spans in an input sequence
    max_mention_length=16,  # maximum length for a single entity span (number of subwords it can span across)
    label_column_name="labels",
    text_column_name="tokens",
    doc_id_column_name="id",
    tokenizer: MLukeTokenizer = None,
    iob_label_list: list = None,
    entity2id: dict = None,
) -> LazyBatch:
    """
    This function accomplishes numerous tasks. First, it determines the total length
    of the input sequence and (if neccessary) splits it into smaller parts. The splitting
    is performed at the sentence boundaries first, and in a second step the function ensures
    that the input sentences do not exceed the maximum sub-word length (`max_num_subwords` keyword argument).

    Next, it finds all possible entity spans (at the token, subword, and character level) as long as they would not exceed the specified maximum
    length (`max_mention_length`). It then maps the ground truth labels (as found in `labels_column_name`)
    to each possible span.

    Finally, each unit is split once more so that the list of possible entity spans does not exceed
    `max_entity_length`. E.g., if `max_entitiy_length` is 32, but there are 100 possible spans, then the
    function will return 4 training units with 25 spans each (evenly distributed). The tokens/subwords
    for each unit are identical, just the spans and their corresponding labels change.

    Input `examples` (LazyBatch): contains following keys:
        id (int): numeric id for example
        tokens (List[str]): tokenized text (probably as provided by the dataset)
        labels (List[int]): NER labels for each token
        sentence_boundaries (List[int]): integers at token level denoting the beginning and end of a sentence (output from `compute_sentence_boundaries_for_luke`)

    Output -> adds the following keys to `examples`:
        `entity_spans` (List[Tuple[int]]): all possible spans for entities at character level (for detokenized text!)
        `text` (str): detokenized text as plain string
        `labels_entity_spans` (List[int]): all correct label IDs for possible entity spans at character level (as found in entity_spans)
        `original_entity_spans` (List[int]): all possible entity spans at the token level **before** tokenization
        `context_boundaries` (List[int]): the cut-off of the text. If an input sequence exceeds the maximum length,
                                          the boundaries keep track of the offset, so that the original input can be restored
                                          and re-aligned with the original labels

    The output of this function increases the size of your dataset!

    The function is based on code found at huggingface and studio ouisa:
    https://github.com/huggingface/transformers/blob/main/examples/research_projects/luke/run_luke_ner_no_trainer.py
    https://github.com/studio-ousia/luke/blob/master/examples/ner/reader.py

    """

    doc_ids = []  # document ids for disentangling predictions later
    texts = []  # raw untokenized text
    all_tokens = []  # tokenized text (preserve for later use)
    all_context_boundaries = (
        []
    )  # when splitting long text inputs, this keeps track on the offsets at the token level
    all_labels = []  # original labels for later use
    all_entity_spans = []  # all possible spans for entities at character level
    all_labels_entity_spans = (
        []
    )  # all correct label IDs for possible entity spans (as found in all_entity_spans)
    all_original_entity_spans = (
        []
    )  # all possible entity spans at the token level (BEFORE tokenization)

    for doc_id, labels, tokens, sentence_boundaries in zip(
        examples[doc_id_column_name],
        examples[label_column_name],
        examples[text_column_name],
        examples["sentence_boundaries"],
    ):

        for s, e in zip(sentence_boundaries[:-1], sentence_boundaries[1:]):
            sentence_words = tokens[s:e]
            subword_lengths = [
                len(tokenizer.tokenize(token)) for token in sentence_words
            ]
            subword2token = list(
                itertools.chain(
                    *[
                        [i] * subword_len
                        for i, subword_len in enumerate(subword_lengths)
                    ]
                )
            )
            # split based on max length (not longer than max_num_subwords (e.g., 126))
            n_splits = math.ceil(sum(subword_lengths) / max_num_subwords)
            # clamp value so it would never exceed max_num_subwords
            # get the longest possible sequence for context to the model
            sequence_length = max(
                min(int(sum(subword_lengths) / n_splits), max_num_subwords),
                max_num_subwords,
            )
            for n in range(n_splits):
                start = n * int(sum(subword_lengths) / n_splits)
                end = start + sequence_length
                if end > sum(subword_lengths):
                    end = sum(subword_lengths)

                if end - start < max_num_subwords:
                    leftover = max_num_subwords - (end - start)
                    start = max(
                        start - leftover, 0
                    )  # ensure it can never be lower than 0

                assert end - start <= max_num_subwords

                _context_subwords = subword2token[start:end]
                # keep track on where the original tokens were sliced
                context_boundaries = (
                    min(_context_subwords),
                    max(_context_subwords) + 1,
                )
                # now we get the perfect slice of context tokens
                context_tokens = tokens[context_boundaries[0] : context_boundaries[1]]
                context_labels = labels[context_boundaries[0] : context_boundaries[1]]
                context_subwords_lengths = subword_lengths[
                    context_boundaries[0] : context_boundaries[1]
                ]

                context = ""
                word_start_char_positions = []
                word_end_char_positions = []
                labels_positions = {}

                tokenindex2wordchar = {}

                # detokenize text, with correct spacing before/after punctuation
                # keep track on the word boundary positions for mapping the NER labels
                for i, word in enumerate(context_tokens):
                    if word[0] == "'" or (len(word) == 1 and is_punctuation(word)):
                        context = context.rstrip()
                    word_start_char_positions.append(len(context))
                    context += word
                    word_end_char_positions.append(len(context))
                    context += " "
                    labels_positions[
                        (word_start_char_positions[-1], word_end_char_positions[-1])
                    ] = entity2id["O"]
                    tokenindex2wordchar[i] = (
                        word_start_char_positions[-1],
                        word_end_char_positions[-1],
                    )

                context = context.rstrip()

                _context_labels = [iob_label_list[l] for l in context_labels]
                for ent in Entities([_context_labels], IOB2).entities[0]:
                    label_start = min(tokenindex2wordchar[ent.start])
                    label_end = max(tokenindex2wordchar[ent.end - 1])
                    labels_positions[(label_start, label_end)] = entity2id[ent.tag]

                entity_spans = []  # spans at the character level
                labels_entity_spans = []  # labels at the character level for each span
                original_entity_spans = []  # spans at the token level

                # find all possible entity spans at the character level
                # assert that the span length does not exceed the maximum length by the model's tokenizer (e.g., 32 subword tokens)
                # also add the corresponding NER tags to the possible spans
                for word_start in range(len(context_tokens)):
                    for word_end in range(word_start, len(context_tokens)):
                        if (
                            sum(context_subwords_lengths[word_start:word_end])
                            <= max_mention_length
                        ):
                            entity_spans.append(
                                (
                                    word_start_char_positions[word_start],
                                    word_end_char_positions[word_end],
                                )
                            )
                            original_entity_spans.append((word_start, word_end + 1))
                            if (
                                word_start_char_positions[word_start],
                                word_end_char_positions[word_end],
                            ) in labels_positions:
                                labels_entity_spans.append(
                                    labels_positions[
                                        (
                                            word_start_char_positions[word_start],
                                            word_end_char_positions[word_end],
                                        )
                                    ]
                                )
                            else:
                                labels_entity_spans.append(entity2id["O"])
                        else:
                            if (
                                word_start_char_positions[word_start],
                                word_end_char_positions[word_end],
                            ) in labels_positions:
                                warnings.warn(
                                    f"An entity was discarded because it exceeded max_mention_length ({max_mention_length}). Span: {context_tokens[word_start:word_end]}"
                                )

                # split into smaller units so that entity_spans does not exceed the length of max_entity_length
                split_size = math.ceil(len(entity_spans) / max_entity_length)
                for i in range(split_size):
                    entity_size = math.ceil(len(entity_spans) / split_size)
                    start = i * entity_size
                    end = start + entity_size
                    all_context_boundaries.append(context_boundaries)
                    all_entity_spans.append(entity_spans[start:end])
                    all_original_entity_spans.append(original_entity_spans[start:end])
                    all_labels_entity_spans.append(labels_entity_spans[start:end])
                    texts.append(context)
                    doc_ids.append(doc_id)
                    all_tokens.append(tokens)
                    all_labels.append(labels)

    examples[doc_id_column_name] = doc_ids
    examples["text"] = texts
    examples["tokens"] = all_tokens
    examples[label_column_name] = all_labels
    examples["entity_spans"] = all_entity_spans
    examples["labels_entity_spans"] = all_labels_entity_spans
    examples["original_entity_spans"] = all_original_entity_spans
    examples["context_boundaries"] = all_context_boundaries

    return examples


def tokenize_and_pad(
    examples: LazyBatch,
    padding="max_length",
    max_length=512,
    doc_id_column_name="id",
    tokenizer: MLukeTokenizer = None,
) -> LazyBatch:
    """
    Tokenize plain strings and also pass all possible entity spans to the tokenizer
    Next, add padding to "labels_entity_spans", "original_entity_spans", and the label_column

    Output with the following keys:
        input_ids (List[int]): sub-word IDs
        entity_ids (List[int]): all zeros for each subword token in input_ids
        entity_position_ids (List[int]): position of each possible entity subword
        entity_start_positions (List[int]): all possible starting positions for entities in the example
        entity_end_positions (List[int]): all possible end positions
        attention_mask (List[int]): attention mask for padding input
        labels (List[int]): ground truth labels for each possible span
        original_entity_spans (List(Tuple[int])): all possible entity spans at token level (before tokenization). Used to get ground truth labels for `compute_metrics`
        labels (List[int]): ground truth NER tag per token (before tokenization). Used to get ground truth labels for `compute_metrics`
    """

    entity_spans = []

    # convert to pairs of tuples
    for v in examples["entity_spans"]:
        entity_spans.append(list(map(tuple, v)))

    tokenized_inputs = tokenizer(
        examples["text"],
        entity_spans=entity_spans,
        max_length=max_length,
        padding=padding,
        truncation=False,
    )

    if padding == "max_length":
        tokenized_inputs["labels"] = padding_tensor(
            examples["labels_entity_spans"],
            -100,
            tokenizer.padding_side,
            tokenizer.max_entity_length,
        )
        tokenized_inputs["original_entity_spans"] = padding_tensor(
            examples["original_entity_spans"],
            (-1, -1),
            tokenizer.padding_side,
            tokenizer.max_entity_length,
        )
    else:
        tokenized_inputs["labels"] = [
            ex[: tokenizer.max_entity_length] for ex in examples["labels_entity_spans"]
        ]
        tokenized_inputs["original_entity_spans"] = [
            ex[: tokenizer.max_entity_length]
            for ex in examples["original_entity_spans"]
        ]

    tokenized_inputs[doc_id_column_name] = examples[doc_id_column_name]
    tokenized_inputs["context_boundaries"] = examples["context_boundaries"]

    return tokenized_inputs


@dataclass
class DataCollatorForLukeTokenClassification(DataCollatorMixin):
    # Class copied from huggingface code:
    # https://github.com/huggingface/transformers/blob/main/examples/research_projects/luke/luke_utils.py

    """

    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~file_utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):
        import torch

        label_name = (
            "processed_labels" if "processed_labels" in features[0].keys() else "labels"
        )
        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["entity_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch[label_name] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label))
                for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label)
                for label in labels
            ]

        original_entity_spans = [
            feature["original_entity_spans"] for feature in features
        ]
        batch["original_entity_spans"] = padding_tensor(
            original_entity_spans, (-1, -1), padding_side, sequence_length
        )
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}

        return batch


def detach_tensors(*tensors: torch.Tensor) -> Iterable[torch.Tensor]:
    """
    Copied from AllenNLP library (published under Apache 2.0 License):
    "If you actually passed gradient-tracking Tensors to a Metric,
    there will be a huge memory leak, because it will prevent garbage
    collection for the computation graph. This method ensures the tensors are detached."

    """
    return (x.detach() if isinstance(x, torch.Tensor) else x for x in tensors)


class SpanToLabelF1:
    """
    Class for taking output by mLUKE and restoring mLUKE's predictions back
    to the standart CoNLL IOB2 format.

    Class keeps track on predictions per document (via doc_id), because we have
    to split each training / evaluation sample into many smaller units.

    Based on original implementation
    https://github.com/studio-ousia/luke/blob/master/examples/ner/metrics/span_to_label_f1.py
    Licensed under Apache 2.0 License (Copyright by Studio-Ousia)

    """

    def __init__(self, id2entity: Dict) -> None:

        self.id2entity = id2entity
        self.predictions = defaultdict(list)
        self.gold_labels = defaultdict(list)

    def __call__(
        self,
        prediction: torch.Tensor,
        gold_labels: torch.Tensor,
        prediction_scores: torch.Tensor,
        original_entity_spans: torch.Tensor,
        doc_id: torch.Tensor,
        context_boundaries: torch.Tensor = None,
    ) -> None:
        """
        Add predictions and ground truth (gold labels) to the metric computation at the end.
        Each call extends the stored predictions and gold labels for each document.
        """

        prediction, gold_labels, prediction_scores, original_entity_spans, doc_id = (
            detach_tensors(
                prediction,
                gold_labels,
                prediction_scores,
                original_entity_spans,
                doc_id,
            )
        )

        if context_boundaries is None:
            context_boundaries = torch.zeros((doc_ids.shape[0], 2), dtype=torch.int8)
        else:
            context_boundaries = context_boundaries.detach()

        for pred, gold, scores, spans, id_, bounds in zip(
            prediction,
            gold_labels,
            prediction_scores,
            original_entity_spans,
            doc_id,
            context_boundaries,
        ):
            pred = pred.tolist()
            gold = gold.tolist()
            scores = scores.tolist()
            offset = bounds.tolist()[0]
            # account for the offset in case a long input text was split into parts
            spans = [(b + offset, e + offset) for b, e in spans.tolist()]
            id_ = int(id_)
            for p, g, score, span in zip(pred, gold, scores, spans):
                if g < 0:
                    continue
                p = self.id2entity[p]
                g = self.id2entity[g]
                self.predictions[id_].append((score, span, p))
                self.gold_labels[id_].append((0, span, g))

    def reset(self) -> None:
        self.predictions = defaultdict(list)
        self.gold_labels = defaultdict(list)

    @staticmethod
    def span_to_label_sequence(
        span_and_labels: List[Tuple[float, Tuple[int, int], str]]
    ) -> List[str]:
        """
        Helper method that greedily selects the best prediction for each span
        in a document.
        This is where the magic happens and the span output of mLUKE is turned into
        CoNLL IOB2 format.
        """
        sequence_length = max([end for score, (start, end), label in span_and_labels])
        label_sequence = ["O"] * sequence_length
        for score, (start, end), label in sorted(span_and_labels, key=lambda x: -x[0]):
            if label == "O" or any([l != "O" for l in label_sequence[start:end]]):
                continue
            label_sequence[start:end] = ["I-" + label] * (end - start)
            label_sequence[start] = "B-" + label
        return label_sequence

    def get_metric(self) -> dict:
        """
        Compute metrics with all samples stored in this instance.
        """

        all_prediction_sequence = []
        all_gold_sequence = []
        for doc_id in self.gold_labels.keys():
            prediction = self.span_to_label_sequence(self.predictions[doc_id])
            gold = self.span_to_label_sequence(self.gold_labels[doc_id])
            all_prediction_sequence.append(prediction)
            all_gold_sequence.append(gold)

        return compute_metrics(all_prediction_sequence, all_gold_sequence)

    def get_luke_labels(self, reset: bool = True) -> Tuple[List[str]]:
        """
        Get the CoNLL IOB2 format for predictions and ground truth labels.

        By default, the storage is `reset`, so all predictions and ground truth labels
        for every document are cleared from the instance.
        """

        all_prediction_sequence = []
        all_gold_sequence = []
        for doc_id in self.gold_labels.keys():
            prediction = self.span_to_label_sequence(self.predictions[doc_id])
            gold = self.span_to_label_sequence(self.gold_labels[doc_id])
            all_prediction_sequence.append(prediction)
            all_gold_sequence.append(gold)

        if reset:
            self.reset()
        return all_prediction_sequence, all_gold_sequence


def evaluate(model: LukeForEntitySpanClassification, dataloader: DataLoader) -> dict:
    """
    Utility function to run an evaluation pass
    on an mLUKE model without a trainer class.

    Probably only works in combination with
    huggingface accelerate.
    """
    span2label = SpanToLabelF1(id2entity)
    model.eval()
    for batch in dataloader:
        original_entity_spans = batch.pop("original_entity_spans")
        doc_ids = batch.pop("doc_id")
        context_boundaries = batch.pop("context_boundaries")
        labels = batch["labels"]
        with torch.no_grad():
            outputs = model(**batch)

        span2label(
            torch.argmax(outputs.logits, axis=2),
            labels,
            torch.max(outputs.logits, axis=2).values,
            original_entity_spans,
            doc_ids,
            context_boundaries=context_boundaries,
        )

    return span2label.get_metric()


if __name__ == "__main__":

    print("Testing is_punctuation")

    for char in [
        ".",
        ":",
        ",",
        ";",
        "'",
        '"',
        "+",
        "!",
        "?",
        "´",
        "`",
        "„",
        "“",
        "‘",
        "’",
        "“",
        "”",
        "…",
    ]:
        assert (
            is_punctuation(char) == True
        ), f"Character: < {char} > is not punctuation!"

    print("Testing pre-processing on CONLL EN dataset")
    from datasets import load_dataset

    raw_datasets = load_dataset("conll2003")

    # Convoluted: we need to restore the labels as string representation
    features = raw_datasets["validation"].features
    iob_label_list = features["ner_tags"].feature.names

    # LUKE does not predict B- or I- tags, it predicts the entity type for the entire span
    # the B- and I- tags are reconstructed later for evaluation
    stripped_labels = sorted(
        list({tag.replace("B-", "").replace("I-", "") for tag in iob_label_list})
    )
    entity2id = {l: i for i, l in enumerate(stripped_labels)}
    id2entity = {v: k for k, v in entity2id.items()}

    print("Casting document ids to int")
    training = raw_datasets["train"]
    training = training.map(lambda x: {"id": int(x["id"])})

    training = compute_sentence_boundaries_for_luke(training[:32])

    print("Initializing tokenizer")
    tokenizer = MLukeTokenizer.from_pretrained(
        "studio-ousia/mluke-base",
        use_fast=False,
        task="entity_span_classification",
        max_entity_length=32,
        max_mention_length=16,
    )

    print("Computing entity spans")
    training = compute_entity_spans_for_luke(
        training,
        **{
            "max_num_subwords": 256,
            "max_entity_length": tokenizer.max_entity_length,
            "max_mention_length": tokenizer.max_mention_length,
            "label_column_name": "ner_tags",  # convert labels to strings on the fly
            "text_column_name": "tokens",
            "doc_id_column_name": "id",
            "tokenizer": tokenizer,
            "entity2id": entity2id,
            "iob_label_list": iob_label_list,
        },
    )

    for col in ["pos_tags", "chunk_tags", "sentence_boundaries"]:
        try:
            _ = training.pop(col)
        except:
            continue

    print("Tokenizing, padding, etc")
    training = tokenize_and_pad(
        training,
        **{"max_length": 256, "tokenizer": tokenizer},
    )

    data_collator = DataCollatorForLukeTokenClassification(
        tokenizer, pad_to_multiple_of=8
    )

    from torch.utils.data import DataLoader
    from datasets import Dataset

    luke_labels = ClassLabel(
        num_classes=len(entity2id.keys()), names=list(entity2id.keys())
    )

    ds = Dataset.from_dict(training)

    data_loader = DataLoader(ds, collate_fn=data_collator, batch_size=8)

    print("Loading batches")
    for batch in data_loader:
        break

    # fake some output
    _shape = list(batch["labels"].shape)
    _shape.append(len(stripped_labels))
    outputs = EntitySpanClassificationOutput(
        loss=torch.rand(1)[0], logits=torch.rand(_shape)
    )

    # prepare data for testing
    original_entity_spans = batch.pop("original_entity_spans")
    ner_tags = batch.pop("ner_tags")
    doc_ids = batch.pop("id")
    labels = batch["labels"]

    metrics = SpanToLabelF1(id2entity)

    metrics(
        torch.argmax(outputs.logits, axis=2),
        labels,
        torch.max(outputs.logits, axis=2).values,
        original_entity_spans,
        doc_ids,
    )
