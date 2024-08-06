# mLUKE

- Citation: Ryokan Ri, Ikuya Yamada, and Yoshimasa Tsuruoka. 2022. mLUKE: The Power of Entity Representations in Multilingual Pretrained Language Models. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7316–7330, Dublin, Ireland. Association for Computational Linguistics. https://aclanthology.org/2022.acl-long.505/
- Model Repository: https://huggingface.co/studio-ousia/mluke-base
- Model Github: https://github.com/studio-ousia/luke

# Fine-tuning

Use `train.py` to fine-tune mLUKE with corpora available in the registry. You can choose the languages with `--languages` and select corpora with the `--corpora` command line arguments.

# Evaluation

Run `evaluate.py` where the first command line argument is the path to the fine-tuned model. You can choose languages and corpora to reduce the number of performed evaluations.