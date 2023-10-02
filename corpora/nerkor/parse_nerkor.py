import sys

from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from utils.registry import add_corpus
from utils.parsers import parse_conllup
from utils.mappings import sonar2conll