from transformers import AutoTokenizer, DataCollatorForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["text"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples[f"labels"]):
        # Map tokens to their respective word.
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            # Only label the first token of a given word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

