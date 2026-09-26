import sys
import argparse
import logging
from bbq.src.common.version import __version__


def run_server_command(args: argparse.Namespace) -> None:
    """Launches the persistent document-indexing server."""
    from bbq.src.terminal.tui import print_bbq

    if not getattr(args, "without_logo", False):
        print_bbq(tag="SERVER", server=getattr(args, "port", 8000))

    from bbq.src.config import Config
    from bbq.src.server import start_document_indexing_server

    config = Config.from_yaml(config_filepath=args.config)
    start_document_indexing_server(
        config=config,
        config_filepath=args.config,
        host=args.host,
        port=args.port,
        without_logo=True,
    )


def run_client_query_command(args: argparse.Namespace) -> None:
    """Launches a client query against the running server with optional Gemini multimodal RAG."""
    # Support checking server status or documents via client query command
    if getattr(args, "status", False):
        run_status_command(args)
        return
    if getattr(args, "documents", False):
        run_documents_command(args)
        return

    from rich.console import Console
    from bbq.src.client import BBQClient
    from bbq.src.config import Config
    from bbq.src.terminal import (
        print_bbq,
        render_query_results_rich,
        render_llm_answer_rich,
        print_interactive_help,
        interactive_page_picker,
    )

    # Load configuration from config.py / config.yaml
    config = Config.from_yaml(config_filepath=getattr(args, "config", "config.yaml"))

    top_k = args.top_k if args.top_k is not None else (config.rag_top_k or 10)
    gemini_key = args.gemini_api_key or config.gemini_api_key
    gemini_model = args.gemini_model or config.gemini_model
    use_llm = getattr(args, "use_llm", False)
    is_infinite = getattr(args, "infinite", False)

    if not getattr(args, "without_logo", False):
        print_bbq(tag="CLIENT", server=args.server, top_k=top_k)

    log_level = logging.INFO if getattr(args, "verbose", False) else logging.WARNING
    logging.basicConfig(
        level=log_level, format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
    )

    console = Console()

    if not args.query and not is_infinite:
        console.print(
            "[red]Error: query text string is required (or run with --infinite for interactive mode).[/red]"
        )
        sys.exit(1)

    client = BBQClient(
        server_url=args.server,
        config=config,
        without_logo=getattr(args, "without_logo", False),
    )

    current_query = args.query

    if is_infinite and not current_query:
        console.print(
            "[dim]Interactive mode active. Enter any query, 'help' for instructions, or 'q' to exit.[/dim]\n"
        )

    while True:
        if not current_query:
            try:
                current_query = console.input(
                    r"[bold red]bbq[/bold red][bold yellow]\[query][/bold yellow] [bold white]>>[/bold white] "
                ).strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Exiting interactive query mode. Bye![/dim]")
                break

        if not current_query:
            if not is_infinite:
                break
            continue

        if current_query.lower() in ("help", "?", ":help"):
            print_interactive_help(console)
            current_query = None
            continue

        if current_query.lower() in ("clear", "cls", ":clear"):
            console.clear()
            print_bbq(tag="CLIENT", server=args.server, top_k=top_k)
            current_query = None
            continue

        if current_query.lower() in ("exit", "quit", ":q", "q"):
            console.print("[dim]Exiting interactive query mode. Bye![/dim]")
            break

        try:
            # 1. Fetch search matches from server
            results = client.query(query_text=current_query, top_k=top_k)

            if not results:
                console.print("[yellow]No matching PDF pages found.[/yellow]")
            else:
                save_images_requested = getattr(args, "save_images", False) or bool(
                    getattr(args, "images_output_dir", None)
                )
                images_out_dir = getattr(args, "images_output_dir", None) or getattr(
                    config, "images_output_dir", "data/rr"
                )

                save_status = None
                save_thread = None
                if save_images_requested and not use_llm:
                    save_status = {
                        "active": True,
                        "completed": False,
                        "saved_count": 0,
                        "total": len(results),
                        "output_dir": images_out_dir,
                        "saved_paths": [],
                    }

                    def _on_progress(idx: int, total: int, path: str) -> None:
                        if path and save_status is not None:
                            save_status["saved_count"] += 1
                            save_status["saved_paths"].append(path)

                    def _on_complete(paths: list) -> None:
                        if save_status is not None:
                            save_status["completed"] = True
                            save_status["active"] = False

                    save_thread = client.save_page_images_threaded(
                        results=results,
                        images_output_dir=images_out_dir,
                        on_progress=_on_progress,
                        on_complete=_on_complete,
                    )

                # 2. Interactive arrow-key page picker (Option C) unless --without-opener is passed
                if not getattr(args, "without_opener", False) and sys.stdin.isatty():
                    interactive_page_picker(results, console=console)
                else:
                    render_query_results_rich(results, console=console)

                if save_images_requested and not use_llm:
                    if save_thread and save_thread.is_alive():
                        if not is_infinite:
                            save_thread.join(timeout=5.0)

                    saved_count = (
                        save_status["saved_count"]
                        if save_status
                        else len([r for r in results if r.get("saved_image_path")])
                    )
                    if saved_count > 0:
                        console.print(
                            f"[bold green]✔ Saved {saved_count} page image(s) to {images_out_dir}[/bold green]\n"
                        )
                    elif save_status and not save_status.get("active"):
                        console.print(
                            f"[bold yellow]⚠ Could not save page images to {images_out_dir}[/bold yellow]\n"
                        )

                # 3. If --use-llm is specified, pass top-k to LLM with cooking spinner
                if use_llm:
                    with console.status(
                        "[bold cyan]LLM model is viewing the query and retrieved pages, cooking the meal...[/bold cyan]",
                        spinner="dots",
                    ):
                        rag_response = client.generate_answer_from_results(
                            query_text=current_query,
                            results=results,
                            gemini_api_key=gemini_key,
                            gemini_model=gemini_model,
                            save_images=save_images_requested,
                            images_output_dir=images_out_dir,
                        )

                    if save_images_requested:
                        saved_paths = [
                            r["saved_image_path"]
                            for r in results
                            if r.get("saved_image_path")
                        ]
                        if saved_paths:
                            console.print(
                                f"[bold green]✔ Saved {len(saved_paths)} page image(s) to {images_out_dir}[/bold green]\n"
                            )

                    answer = rag_response.get("answer")
                    if answer:
                        render_llm_answer_rich(
                            answer=answer,
                            engine=rag_response.get("engine", "gemini"),
                            console=console,
                        )
                    else:
                        fallback_reason = rag_response.get(
                            "fallback_reason", "No API key or API call failed"
                        )
                        console.print(
                            f"\n[yellow][Note: LLM answer unavailable: {fallback_reason}][/yellow]"
                        )

        except Exception as err:
            console.print(f"[red]Error querying BBQ server:[/red] {err}")
            if not is_infinite:
                sys.exit(1)

        if not is_infinite:
            break

        current_query = None


def run_status_command(args: argparse.Namespace) -> None:
    """Fetches and displays running server status."""
    from bbq.src.client import BBQClient
    from bbq.src.terminal.tui import print_bbq

    if not getattr(args, "without_logo", False):
        print_bbq(tag="CLIENT", server=args.server)

    client = BBQClient(
        server_url=args.server, without_logo=getattr(args, "without_logo", False)
    )
    try:
        status = client.get_status()
        print("Server Status:")
        print("=" * 40)
        for k, v in status.items():
            print(f"  {k:<20}: {v}")
    except Exception as err:
        print(f"Error fetching server status: {err}", file=sys.stderr)
        sys.exit(1)


def run_documents_command(args: argparse.Namespace) -> None:
    """Lists all indexed document records."""
    from bbq.src.client import BBQClient
    from bbq.src.terminal.tui import print_bbq

    if not getattr(args, "without_logo", False):
        print_bbq(tag="CLIENT", server=args.server)

    client = BBQClient(
        server_url=args.server, without_logo=getattr(args, "without_logo", False)
    )
    try:
        documents = client.list_documents()
        if not documents:
            print("No indexed documents found.")
            return
        print(f"Indexed Documents ({len(documents)} total):")
        print("=" * 60)
        for doc in documents:
            print(f"  File Hash : {doc.get('file_hash', '')[:12]}")
            print(f"  File Path : {doc.get('file_path')}")
            print(f"  Status    : {doc.get('status')}")
            print(f"  Pages     : {doc.get('num_pages')}")
            print("-" * 60)
    except Exception as err:
        print(f"Error fetching documents: {err}", file=sys.stderr)
        sys.exit(1)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bbq",
        description="BBQ RAG - Unified Server & Client CLI Engine",
    )
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=f"bbq {__version__}",
    )
    parser.add_argument(
        "--without-logo",
        "--no-logo",
        action="store_true",
        default=False,
        help="Do not print BBQ logo banner on run",
    )

    client_parent = argparse.ArgumentParser(add_help=False)
    client_parent.add_argument(
        "--without-logo",
        "--no-logo",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Do not print BBQ logo banner on run",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Server Command ---
    server_parser = subparsers.add_parser(
        "server",
        parents=[client_parent],
        help="Start persistent document indexing HTTP server",
    )
    server_parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    server_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="HTTP server bind host (default: 0.0.0.0)",
    )
    server_parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8000,
        help="HTTP server bind port (default: 8000)",
    )
    server_parser.set_defaults(func=run_server_command)

    # --- Query / Client Command ---
    query_parser = subparsers.add_parser(
        "query",
        aliases=["client"],
        parents=[client_parent],
        help="Query indexed documents via client API with optional Gemini RAG",
    )
    query_parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default=None,
        help="Search query text string (optional when using --infinite)",
    )
    query_parser.add_argument(
        "--infinite",
        "-inf",
        action="store_true",
        default=False,
        help="Run interactive continuous query prompt loop",
    )
    query_parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    query_parser.add_argument(
        "--server",
        "-s",
        type=str,
        default="http://localhost:8000",
        help="Server URL (default: http://localhost:8000)",
    )
    query_parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=None,
        help="Top K results to retrieve (default from config.yaml rag_top_k: 20)",
    )
    query_parser.add_argument(
        "--use-llm",
        action="store_true",
        default=False,
        help="Pass top-k retrieved pages to LLM for multimodal answer generation",
    )
    query_parser.add_argument(
        "--gemini-api-key",
        "-g",
        type=str,
        default=None,
        help="Google Gemini API key (or set in config.yaml / GEMINI_API_KEY env var)",
    )
    query_parser.add_argument(
        "--gemini-model",
        type=str,
        default=None,
        help="Gemini model name (default from config: gemini-3.6-flash)",
    )
    query_parser.add_argument(
        "--save-images",
        "-i",
        action="store_true",
        help="Save page images to disk (default output: data/rr)",
    )
    query_parser.add_argument(
        "--images-output-dir",
        "-o",
        type=str,
        default=None,
        help="Directory to save page images (default from config: data/rr)",
    )
    query_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose client debug logging",
    )
    query_parser.add_argument(
        "--without-opener",
        "--no-opener",
        action="store_true",
        default=False,
        help="Do not launch interactive document page opener after search query",
    )
    query_parser.set_defaults(func=run_client_query_command)

    # --- Status Command ---
    status_parser = subparsers.add_parser(
        "status", parents=[client_parent], help="Check running server status"
    )
    status_parser.add_argument(
        "--server", "-s", type=str, default="http://localhost:8000", help="Server URL"
    )
    status_parser.set_defaults(func=run_status_command)

    # --- Documents Command ---
    docs_parser = subparsers.add_parser(
        "documents", parents=[client_parent], help="List indexed document metadata"
    )
    docs_parser.add_argument(
        "--server", "-s", type=str, default="http://localhost:8000", help="Server URL"
    )
    docs_parser.set_defaults(func=run_documents_command)

    return parser


def build_client_main_parser() -> argparse.ArgumentParser:
    """Builds the standalone parser used by python -m bbq.src.client."""
    parser = argparse.ArgumentParser(
        prog="python -m bbq.src.client",
        description="BBQ RAG - Client CLI Query & Management Engine",
    )
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=f"bbq {__version__}",
    )
    parser.add_argument(
        "query", type=str, nargs="?", default=None, help="Search query text string"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--server",
        "-s",
        type=str,
        default="http://localhost:8000",
        help="Server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=None,
        help="Top K results to retrieve (default from config.yaml rag_top_k: 20)",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        default=False,
        help="Pass top-k retrieved pages to LLM for multimodal answer generation",
    )
    parser.add_argument(
        "--infinite",
        "-inf",
        action="store_true",
        default=False,
        help="Run interactive continuous query prompt loop",
    )
    parser.add_argument(
        "--gemini-api-key", "-g", type=str, default=None, help="Google Gemini API key"
    )
    parser.add_argument(
        "--gemini-model", type=str, default=None, help="Gemini model name"
    )
    parser.add_argument(
        "--save-images",
        "-i",
        action="store_true",
        help="Save page images to disk (default output: data/rr)",
    )
    parser.add_argument(
        "--images-output-dir",
        "-o",
        type=str,
        default=None,
        help="Directory to save page images (default from config: data/rr)",
    )
    parser.add_argument(
        "--without-logo",
        "--no-logo",
        action="store_true",
        default=False,
        help="Do not print BBQ logo banner on run",
    )
    parser.add_argument(
        "--without-opener",
        "--no-opener",
        action="store_true",
        default=False,
        help="Do not launch interactive document page opener",
    )
    parser.add_argument("--status", action="store_true", help="Check server status")
    parser.add_argument(
        "--documents", action="store_true", help="List indexed documents"
    )
    return parser


def main() -> None:
    from bbq.src.common.hardware import verify_hardware_or_exit

    verify_hardware_or_exit()

    # Backward compatibility fallback: if run as 'python -m bbq.src.main [config.yaml]' without subcommands
    if (
        len(sys.argv) > 1
        and not sys.argv[1].startswith("-")
        and sys.argv[1]
        not in [
            "server",
            "query",
            "client",
            "status",
            "documents",
            "-h",
            "--help",
            "-V",
            "--version",
        ]
    ):
        config_path = sys.argv[1]
        print(f"Launching server mode with config: {config_path}")
        from bbq.src.server import start_document_indexing_server

        start_document_indexing_server(config_filepath=config_path)
        return

    parser = build_argument_parser()

    # Default to server mode if no arguments provided
    if len(sys.argv) == 1:
        from bbq.src.server import start_document_indexing_server

        start_document_indexing_server(config_filepath="config.yaml")
        return

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
