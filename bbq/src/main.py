import sys
import os
import argparse
import logging
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def run_server_command(args: argparse.Namespace) -> None:
    """Launches the persistent document-indexing server."""
    from bbq.src.server import start_document_indexing_server
    start_document_indexing_server(
        config_filepath=args.config,
        host=args.host,
        port=args.port,
    )


def run_client_query_command(args: argparse.Namespace) -> None:
    """Launches a client query against the running server with optional Gemini multimodal RAG."""
    from bbq.src.client import BBQClient
    from bbq.src.config import load_configuration_from_yaml_file

    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")

    # Load configuration from config.py / config.yaml
    config = load_configuration_from_yaml_file(config_filepath=args.config)

    top_k = args.top_k if args.top_k is not None else config.rag_top_k
    gemini_key = args.gemini_api_key or config.gemini_api_key
    gemini_model = args.gemini_model or config.gemini_model

    client = BBQClient(server_url=args.server, config=config)
    try:
        print(f"Sending query to {args.server}: '{args.query}' (top_k={top_k})...\n")

        # Execute unified query + multimodal answer generation
        rag_response = client.query_and_answer(
            query_text=args.query,
            top_k=top_k,
            gemini_api_key=gemini_key,
            gemini_model=gemini_model,
            save_images=args.save_images,
        )

        results = rag_response.get("sources", [])
        answer = rag_response.get("answer")
        status = rag_response.get("status")

        if not results:
            print("No matching PDF pages found.")
            return

        # 1. If Gemini generated an answer, display it prominently
        if answer:
            print("=" * 60)
            print(f"GEMINI MULTIMODAL ANSWER ({rag_response.get('engine', 'gemini')}):")
            print("=" * 60)
            print(answer)
            print("\n" + "=" * 60)
            print(f"GROUNDED RETRIEVED SOURCES (Top {len(results)} Pages):")
            print("=" * 60)
        else:
            if status == "fallback_pages_only":
                reason = rag_response.get("fallback_reason", "No API key or API call failed")
                print(f"[Note: {reason}]")
            print(f"Top {len(results)} Matching PDF Pages (Book Results):\n" + "=" * 60)

        # 2. Display the retrieved book pages
        for i, res in enumerate(results, 1):
            print(f"Rank {i}:")
            print(f"  Score       : {res['score']:.4f}")
            print(f"  PDF File    : {res['file_path']}")
            print(f"  Page        : Page {res['page_number']} of {res['total_pages']}")
            print(f"  File Hash   : {res['file_hash'][:12]}")
            if res.get("saved_image_path"):
                print(f"  Saved Image : {res['saved_image_path']}")
            print("-" * 60)

    except Exception as err:
        print(f"Error querying BBQ server: {err}", file=sys.stderr)
        sys.exit(1)


def run_status_command(args: argparse.Namespace) -> None:
    """Fetches and displays running server status."""
    from bbq.src.client import BBQClient

    client = BBQClient(server_url=args.server)
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

    client = BBQClient(server_url=args.server)
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

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Server Command ---
    server_parser = subparsers.add_parser("server", help="Start persistent document indexing HTTP server")
    server_parser.add_argument(
        "--config", "-c", type=str, default="config.yaml", help="Path to YAML config file (default: config.yaml)"
    )
    server_parser.add_argument("--host", type=str, default="0.0.0.0", help="HTTP server bind host (default: 0.0.0.0)")
    server_parser.add_argument("--port", "-p", type=int, default=8000, help="HTTP server bind port (default: 8000)")
    server_parser.set_defaults(func=run_server_command)

    # --- Query / Client Command ---
    query_parser = subparsers.add_parser("query", help="Query indexed documents via client API with optional Gemini RAG")
    query_parser.add_argument("query", type=str, help="Search query text string")
    query_parser.add_argument(
        "--config", "-c", type=str, default="config.yaml", help="Path to YAML config file (default: config.yaml)"
    )
    query_parser.add_argument("--server", "-s", type=str, default="http://localhost:8000", help="Server URL (default: http://localhost:8000)")
    query_parser.add_argument("--top-k", "-k", type=int, default=None, help="Top K results to retrieve (default from config: 3)")
    query_parser.add_argument(
        "--gemini-api-key", "-g", type=str, default=None, help="Google Gemini API key (or set in config.yaml / GEMINI_API_KEY env var)"
    )
    query_parser.add_argument(
        "--gemini-model", type=str, default=None, help="Gemini model name (default from config: gemini-3.6-flash)"
    )
    query_parser.add_argument("--save-images", "-i", action="store_true", help="Save page images to disk")
    query_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose client debug logging")
    query_parser.set_defaults(func=run_client_query_command)

    # --- Status Command ---
    status_parser = subparsers.add_parser("status", help="Check running server status")
    status_parser.add_argument("--server", "-s", type=str, default="http://localhost:8000", help="Server URL")
    status_parser.set_defaults(func=run_status_command)

    # --- Documents Command ---
    docs_parser = subparsers.add_parser("documents", help="List indexed document metadata")
    docs_parser.add_argument("--server", "-s", type=str, default="http://localhost:8000", help="Server URL")
    docs_parser.set_defaults(func=run_documents_command)

    return parser


def main() -> None:
    # Backward compatibility fallback: if run as 'python -m bbq.src.main [config.yaml]' without subcommands
    if (
        len(sys.argv) > 1
        and not sys.argv[1].startswith("-")
        and sys.argv[1] not in ["server", "query", "status", "documents", "-h", "--help"]
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
