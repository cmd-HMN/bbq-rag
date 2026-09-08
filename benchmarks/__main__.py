import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.cli import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        from rich.console import Console
        Console().print("[bold red]Closing program, the file might be corrupted.[/bold red]")
        exit(1)
