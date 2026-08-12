import sys
from pathlib import Path

# Ensure bbq root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bbq.src.server import start_document_indexing_server


def main() -> None:
    config_filepath: str = "config.yaml"
    if len(sys.argv) > 1:
        config_filepath = sys.argv[1]
    start_document_indexing_server(config_filepath=config_filepath)


if __name__ == "__main__":
    main()
