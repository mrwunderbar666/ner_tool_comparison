from typing import Optional, List
from seqeval.metrics import classification_report, accuracy_score

def compute_metrics(
        predictions: list,
        references: list,
        suffix: bool = False,
        sample_weight: Optional[List[int]] = None) -> dict:
    
    report = classification_report(
        y_true=references,
        y_pred=predictions,
        suffix=suffix,
        output_dict=True,
        sample_weight=sample_weight,
        zero_division=0
    )
    report.pop("macro avg")
    report.pop("weighted avg")
    overall_score = report.pop("micro avg")

    scores = {
        type_name: {
            "precision": score["precision"],
            "recall": score["recall"],
            "f1": score["f1-score"],
            "number": score["support"],
        }
        for type_name, score in report.items()
    }
    scores["overall_precision"] = overall_score["precision"]
    scores["overall_recall"] = overall_score["recall"]
    scores["overall_f1"] = overall_score["f1-score"]
    scores["overall_accuracy"] = accuracy_score(y_true=references, y_pred=predictions)

    return scores


if __name__ == "__main__":
    print('testing compute_metrics function')
    y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]

    result = compute_metrics(y_pred, y_true)
    assert result['overall_f1'] == 0.50, print(result)
    print('passed')