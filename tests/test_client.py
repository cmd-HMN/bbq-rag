"""
Unit tests for BBQ RAG Client module, Gemini client, logo banner printing,
and --without-logo suppression behavior.
"""

import io
import os
from unittest.mock import patch, MagicMock
from PIL import Image

import pytest
from bbq.src.terminal.tui import print_bbq
from bbq.src.client import BBQClient, GeminiClient
from bbq.src.main import (
    build_argument_parser,
    build_client_main_parser,
    __version__,
)


def test_print_bbq_banner_output(capsys):
    """Verify that print_bbq prints the ASCII banner with BBQ, tag, server, and top-k annotations."""
    print_bbq(name="BBQ", tag="CLIENT", without_logo=False)
    captured = capsys.readouterr().out
    assert "BBQ" in captured
    assert "[CLIENT]" in captured

    print_bbq(tag="CLIENT", server="http://localhost:8080", top_k=4)
    captured2 = capsys.readouterr().out
    assert "BBQ" in captured2
    assert "[CLIENT]" in captured2
    assert "[8080]" in captured2
    assert "[top-4]" in captured2


def test_print_bbq_without_logo_suppresses_output(capsys):
    """Verify that print_bbq prints nothing when without_logo is True."""
    print_bbq(name="BBQ", tag="CLIENT", without_logo=True)
    captured = capsys.readouterr().out
    assert captured == ""


def test_client_init_and_attributes():
    """Verify BBQClient attributes and initialization."""
    client = BBQClient(server_url="http://test-server:9000/", without_logo=True)
    assert client.server_url == "http://test-server:9000"
    assert client.without_logo is True

    default_client = BBQClient()
    assert default_client.server_url == "http://localhost:8000"
    assert default_client.without_logo is False


def test_client_print_logo(capsys):
    """Verify BBQClient.print_logo respects without_logo."""
    client_with_logo = BBQClient(without_logo=False)
    client_with_logo.print_logo()
    out = capsys.readouterr().out
    assert "CLIENT" in out

    client_without_logo = BBQClient(without_logo=True)
    client_without_logo.print_logo()
    assert capsys.readouterr().out == ""


@patch("requests.post")
def test_client_query_mock(mock_post):
    """Verify client.query posts to server endpoint."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [{"file_path": "doc.pdf", "page_number": 1, "score": 2.5}]
    }
    mock_post.return_value = mock_response

    client = BBQClient(server_url="http://localhost:8000")
    results = client.query("test query", top_k=5)

    assert len(results) == 1
    assert results[0]["file_path"] == "doc.pdf"
    assert results[0]["page_number"] == 1
    mock_post.assert_called_once_with(
        "http://localhost:8000/query",
        json={"query": "test query", "top_k": 5},
        timeout=30,
    )


@patch("requests.get")
def test_client_status_and_documents_mock(mock_get):
    """Verify client.get_status and client.list_documents."""
    mock_status_resp = MagicMock()
    mock_status_resp.json.return_value = {"status": "online", "indexed_documents": 10}
    mock_docs_resp = MagicMock()
    mock_docs_resp.json.return_value = {"documents": [{"file_hash": "abc"}]}

    mock_get.side_effect = [mock_status_resp, mock_docs_resp]

    client = BBQClient()
    status = client.get_status()
    assert status["status"] == "online"

    docs = client.list_documents()
    assert len(docs) == 1
    assert docs[0]["file_hash"] == "abc"


@patch("requests.get")
def test_client_get_page_image_mock(mock_get, tmp_path):
    """Verify client.get_page_image retrieves image bytes and optionally saves to disk."""
    # Create simple dummy PNG image bytes
    img = Image.new("RGB", (10, 10), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    mock_resp = MagicMock()
    mock_resp.content = png_bytes
    mock_get.return_value = mock_resp

    client = BBQClient()
    save_file = str(tmp_path / "page_1.png")
    ret_img = client.get_page_image("test.pdf", page_number=1, save_path=save_file)

    assert isinstance(ret_img, Image.Image)
    assert (tmp_path / "page_1.png").exists()


def test_gemini_client_availability():
    """Verify GeminiClient availability check."""
    client_no_key = GeminiClient(api_key="")
    assert client_no_key.is_available() is False

    client_with_key = GeminiClient(api_key="valid-key-xyz")
    assert client_with_key.is_available() is True
    assert client_with_key.model == "gemini-3.6-flash"


def test_main_cli_without_logo_options():
    """Verify that CLI parser flags --without-logo and --no-logo are correctly recognized."""
    parser = build_argument_parser()

    # Flag before query command
    args1 = parser.parse_args(["--without-logo", "query", "what is this?"])
    assert args1.without_logo is True
    assert args1.query == "what is this?"

    # Flag after query command
    args2 = parser.parse_args(["query", "--without-logo", "what is this?"])
    assert args2.without_logo is True

    # --no-logo alias
    args3 = parser.parse_args(["query", "--no-logo", "what is this?"])
    assert args3.without_logo is True

    # Default without flag
    args4 = parser.parse_args(["query", "what is this?"])
    assert args4.without_logo is False

    # Status command with --without-logo
    args5 = parser.parse_args(["status", "--without-logo"])
    assert args5.without_logo is True

    # Client alias command
    args6 = parser.parse_args(["client", "find documents"])
    assert args6.query == "find documents"


def test_client_main_cli_parser():
    """Verify that bbq.src.client.__main__ parser recognizes flags."""
    parser = build_client_main_parser()
    args = parser.parse_args(["my search", "--without-logo", "--top-k", "4"])
    assert args.query == "my search"
    assert args.without_logo is True
    assert args.top_k == 4


def test_cli_use_llm_and_default_top_k():
    """Verify --use-llm flag and default top_k=None in parsers (deferring to config.rag_top_k)."""
    # Main query parser
    main_parser = build_argument_parser()
    args_default = main_parser.parse_args(["query", "search text"])
    assert args_default.top_k is None
    assert args_default.use_llm is False

    args_explicit = main_parser.parse_args(["query", "--top-k", "15", "search text"])
    assert args_explicit.top_k == 15

    args_llm = main_parser.parse_args(["query", "--use-llm", "search text"])
    assert args_llm.use_llm is True

    # Client __main__ parser
    client_parser = build_client_main_parser()
    c_args_default = client_parser.parse_args(["search text"])
    assert c_args_default.top_k is None
    assert c_args_default.use_llm is False

    c_args_explicit = client_parser.parse_args(["search text", "--top-k", "5"])
    assert c_args_explicit.top_k == 5

    c_args_llm = client_parser.parse_args(["search text", "--use-llm"])
    assert c_args_llm.use_llm is True


@patch.object(BBQClient, "query")
def test_client_query_and_answer_without_llm(mock_query):
    """Verify that query_and_answer with use_llm=False returns retrieved results without calling Gemini."""
    mock_query.return_value = [{"file_path": "doc.pdf", "page_number": 1, "score": 3.0}]
    client = BBQClient()

    resp = client.query_and_answer("query text", top_k=10, use_llm=False)
    assert resp["status"] == "retrieval_only"
    assert resp["answer"] is None
    assert len(resp["sources"]) == 1
    mock_query.assert_called_once_with(query_text="query text", top_k=10)


@patch.object(BBQClient, "generate_answer_from_results")
@patch.object(BBQClient, "query")
def test_client_query_and_answer_with_llm(mock_query, mock_generate):
    """Verify that query_and_answer with use_llm=True calls generate_answer_from_results."""
    mock_query.return_value = [{"file_path": "doc.pdf", "page_number": 1, "score": 3.0}]
    mock_generate.return_value = {
        "query": "query text",
        "answer": "Generated multimodal answer",
        "sources": [{"file_path": "doc.pdf", "page_number": 1, "score": 3.0}],
        "status": "success",
        "engine": "gemini",
    }
    client = BBQClient()

    resp = client.query_and_answer("query text", top_k=10, use_llm=True)
    assert resp["status"] == "success"
    assert resp["answer"] == "Generated multimodal answer"
    mock_generate.assert_called_once()


def test_cli_infinite_flags_parsing():
    """Verify --infinite and -inf flag parsing in both main and client __main__ parsers."""
    main_parser = build_argument_parser()

    args1 = main_parser.parse_args(["query", "--infinite"])
    assert args1.infinite is True
    assert args1.query is None

    args2 = main_parser.parse_args(["client", "-inf", "initial query"])
    assert args2.infinite is True
    assert args2.query == "initial query"

    client_parser = build_client_main_parser()
    c_args1 = client_parser.parse_args(["--infinite"])
    assert c_args1.infinite is True
    assert c_args1.query is None

    c_args2 = client_parser.parse_args(["-inf", "first query"])
    assert c_args2.infinite is True
    assert c_args2.query == "first query"


@patch("builtins.input")
@patch.object(BBQClient, "query")
def test_main_cli_infinite_mode_interactive_loop(mock_query, mock_input, capsys):
    """Verify interactive loop runs with initial query and prompts for next until exit."""
    from bbq.src.main import run_client_query_command

    mock_query.return_value = [
        {"file_path": "doc.pdf", "page_number": 1, "score": 2.5, "file_hash": "hash123"}
    ]
    # Simulate user input after the initial query finishes: first another query, then 'exit'
    mock_input.side_effect = ["second query", "exit"]

    parser = build_argument_parser()
    args = parser.parse_args(
        ["query", "first query", "--top-k", "10", "--infinite", "--without-logo"]
    )
    run_client_query_command(args)

    # query should have been called twice: first query, then second query
    assert mock_query.call_count == 2
    mock_query.assert_any_call(query_text="first query", top_k=10)
    mock_query.assert_any_call(query_text="second query", top_k=10)

    out = capsys.readouterr().out
    assert "Exiting interactive query mode." in out


@patch("builtins.input")
@patch.object(BBQClient, "query")
def test_main_cli_infinite_mode_keyboard_interrupt(mock_query, mock_input, capsys):
    """Verify interactive loop terminates gracefully on KeyboardInterrupt."""
    from bbq.src.main import run_client_query_command

    mock_input.side_effect = KeyboardInterrupt()

    parser = build_argument_parser()
    # No initial query provided, goes straight to input prompt
    args = parser.parse_args(["query", "--infinite", "--without-logo"])
    run_client_query_command(args)

    mock_query.assert_not_called()
    out = capsys.readouterr().out
    assert "Exiting interactive query mode." in out


def test_cli_version_flag(capsys):
    """Verify that --version dynamically outputs the package version and exits."""
    parser = build_argument_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr().out
    assert __version__ in captured
    assert "bbq" in captured


@patch("builtins.input")
def test_main_cli_interactive_help_command(mock_input, capsys):
    """Verify that entering 'help' displays the interactive commands menu."""
    from bbq.src.main import run_client_query_command

    mock_input.side_effect = ["help", "q"]

    parser = build_argument_parser()
    args = parser.parse_args(["query", "--infinite", "--without-logo"])
    run_client_query_command(args)

    out = capsys.readouterr().out
    assert "bbq[query] Interactive Commands" in out
    assert "Enter any query" in out
    assert "q to exit" in out or "quit" in out


def test_render_rich_components_no_tables(capsys):
    """Verify render_query_results_rich and render_llm_answer_rich render without crashing."""
    from bbq.src.terminal import render_query_results_rich, render_llm_answer_rich

    sample_results = [
        {
            "score": 5.7152,
            "file_path": "data/watch/puzzles.pdf",
            "page_number": 7,
            "total_pages": 98,
            "file_hash": "0cf7e26873a3",
        }
    ]
    render_query_results_rich(sample_results)
    out = capsys.readouterr().out
    assert "Top 1 Matching PDF Pages" in out
    assert "Page" in out and "7" in out
    assert "puzzles.pdf" in out

    render_llm_answer_rich("This is a synthesized multimodal answer.")
    llm_out = capsys.readouterr().out
    assert "GEMINI MULTIMODAL ANSWER" in llm_out
    assert "synthesized multimodal answer" in llm_out


@patch("builtins.input")
@patch("rich.console.Console.clear")
def test_main_cli_interactive_clear_command(mock_clear, mock_input, capsys):
    """Verify that entering 'clear' or 'cls' triggers console.clear()."""
    from bbq.src.main import run_client_query_command

    mock_input.side_effect = ["clear", "cls", "q"]

    parser = build_argument_parser()
    args = parser.parse_args(["query", "--infinite", "--without-logo"])
    run_client_query_command(args)

    assert mock_clear.call_count == 2
    out = capsys.readouterr().out
    assert "Exiting interactive query mode. Bye!" in out


@pytest.mark.skipif(
    bool(os.environ.get("CI")) or bool(os.environ.get("GITHUB_ACTIONS")),
    reason="Document page opener requires GUI PDF viewer not available in headless CI environment",
)
def test_open_document_page(monkeypatch):
    """Verify open_document_page calls subprocess.Popen with page arguments."""
    from bbq.src.terminal import open_document_page

    mock_calls = []

    def mock_popen(args, **kwargs):
        mock_calls.append(args)
        return MagicMock()

    monkeypatch.setattr("subprocess.Popen", mock_popen)
    monkeypatch.setattr("os.path.exists", lambda p: True)
    monkeypatch.setattr("shutil.which", lambda cmd: f"/usr/bin/{cmd}")

    success = open_document_page("data/watch/test.pdf", page_number=42)
    assert success is True
    assert len(mock_calls) == 1
    # Check that page argument was passed
    call_args = mock_calls[0]
    assert "42" in call_args


def test_interactive_page_picker_non_tty(capsys):
    """Verify interactive_page_picker renders cleanly when stdin is not a tty."""
    from bbq.src.terminal import interactive_page_picker

    sample_results = [
        {
            "score": 4.5,
            "file_path": "data/watch/design.pdf",
            "page_number": 128,
            "total_pages": 241,
        }
    ]
    interactive_page_picker(sample_results)
    out = capsys.readouterr().out
    assert "Top 1 Matching PDF Pages" in out
    assert "design.pdf" in out


def test_without_opener_flag(monkeypatch, capsys):
    """Verify that --without-opener bypasses interactive_page_picker and prints static results."""
    from bbq.src.main import run_client_query_command
    from bbq.src.client import BBQClient

    mock_picker_called = []

    monkeypatch.setattr(
        BBQClient,
        "query",
        lambda self, query_text, top_k=10: [
            {"file_path": "doc.pdf", "page_number": 5, "total_pages": 10, "score": 3.0}
        ],
    )
    monkeypatch.setattr(
        "bbq.src.terminal.interactive_page_picker",
        lambda results, console=None: mock_picker_called.append(True),
    )

    parser = build_argument_parser()
    args = parser.parse_args(["query", "search", "--without-opener", "--without-logo"])
    assert args.without_opener is True
    run_client_query_command(args)

    assert len(mock_picker_called) == 0
    out = capsys.readouterr().out
    assert "Top 1 Matching PDF Pages" in out


@patch.object(BBQClient, "get_page_image")
def test_client_save_page_images(mock_get_img, tmp_path):
    """Verify BBQClient.save_page_images saves page images with proper filenames to disk."""
    dummy_img = Image.new("RGB", (20, 20), color="red")

    def fake_get_page_image(file_path, page_number=1, dpi=None, save_path=None):
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            dummy_img.save(save_path, format="PNG")
        return dummy_img

    mock_get_img.side_effect = fake_get_page_image

    client = BBQClient()
    results = [
        {"file_path": "docA.pdf", "page_number": 12, "score": 4.1},
        {"file_path": "docB.pdf", "page_number": 45, "score": 3.2},
    ]
    out_dir = str(tmp_path / "saved_rr")
    saved_paths = client.save_page_images(results, images_output_dir=out_dir)

    assert len(saved_paths) == 2
    assert os.path.exists(os.path.join(out_dir, "rr_1_12.png"))
    assert os.path.exists(os.path.join(out_dir, "rr_2_45.png"))
    assert results[0]["image_available"] is True
    assert results[0]["saved_image_path"] == os.path.join(out_dir, "rr_1_12.png")
    assert results[1]["saved_image_path"] == os.path.join(out_dir, "rr_2_45.png")


@patch.object(BBQClient, "query")
@patch.object(BBQClient, "get_page_image")
def test_client_query_and_answer_save_images_without_llm(
    mock_get_img, mock_query, tmp_path
):
    """Verify query_and_answer with use_llm=False and save_images=True saves images without calling LLM."""
    dummy_img = Image.new("RGB", (10, 10), color="green")

    def fake_get_page_image(file_path, page_number=1, dpi=None, save_path=None):
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            dummy_img.save(save_path, format="PNG")
        return dummy_img

    mock_get_img.side_effect = fake_get_page_image
    mock_query.return_value = [
        {"file_path": "manual.pdf", "page_number": 7, "score": 2.9}
    ]

    out_dir = str(tmp_path / "retrieval_images")
    client = BBQClient()
    resp = client.query_and_answer(
        "search terms",
        top_k=5,
        use_llm=False,
        save_images=True,
        images_output_dir=out_dir,
    )

    assert resp["status"] == "retrieval_only"
    assert resp["answer"] is None
    assert os.path.exists(os.path.join(out_dir, "rr_1_7.png"))


def test_cli_save_images_flags_parsing():
    """Verify parsing of --save-images, -i, and --images-output-dir / -o flags."""
    parser = build_argument_parser()

    args1 = parser.parse_args(["query", "search", "--save-images"])
    assert args1.save_images is True

    args2 = parser.parse_args(["client", "search", "-i"])
    assert args2.save_images is True

    args3 = parser.parse_args(["query", "search", "-i", "-o", "custom/dir"])
    assert args3.save_images is True
    assert args3.images_output_dir == "custom/dir"

    client_parser = build_client_main_parser()
    c_args1 = client_parser.parse_args(["search", "-i"])
    assert c_args1.save_images is True

    c_args2 = client_parser.parse_args(["search", "--images-output-dir", "custom/path"])
    assert c_args2.images_output_dir == "custom/path"


@patch.object(BBQClient, "query")
@patch.object(BBQClient, "get_page_image")
def test_cli_save_images_execution(mock_get_img, mock_query, tmp_path, capsys):
    """Verify CLI saves images to specified directory when -i is passed without --use-llm."""
    from bbq.src.main import run_client_query_command

    dummy_img = Image.new("RGB", (10, 10), color="yellow")

    def fake_get_page_image(file_path, page_number=1, dpi=None, save_path=None):
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            dummy_img.save(save_path, format="PNG")
        return dummy_img

    mock_get_img.side_effect = fake_get_page_image
    mock_query.return_value = [
        {"file_path": "doc.pdf", "page_number": 3, "total_pages": 10, "score": 3.0}
    ]

    out_dir = str(tmp_path / "cli_rr")
    parser = build_argument_parser()
    args = parser.parse_args(
        [
            "query",
            "test save",
            "-i",
            "-o",
            out_dir,
            "--without-opener",
            "--without-logo",
        ]
    )
    run_client_query_command(args)

    assert os.path.exists(os.path.join(out_dir, "rr_1_3.png"))
    captured = capsys.readouterr().out
    assert "Saved 1 page image(s)" in captured


@patch.object(BBQClient, "get_page_image")
def test_client_save_page_images_threaded(mock_get_img, tmp_path):
    """Verify BBQClient.save_page_images_threaded executes in a background thread and fires callbacks."""
    import threading

    dummy_img = Image.new("RGB", (10, 10), color="purple")

    def fake_get_page_image(file_path, page_number=1, dpi=None, save_path=None):
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            dummy_img.save(save_path, format="PNG")
        return dummy_img

    mock_get_img.side_effect = fake_get_page_image

    progress_events = []
    complete_events = []

    def on_prog(idx, total, path):
        progress_events.append((idx, total, path))

    def on_comp(paths):
        complete_events.append(paths)

    client = BBQClient()
    results = [
        {"file_path": "ch1.pdf", "page_number": 5, "score": 4.0},
        {"file_path": "ch2.pdf", "page_number": 9, "score": 3.8},
    ]
    out_dir = str(tmp_path / "threaded_rr")
    thread = client.save_page_images_threaded(
        results=results,
        images_output_dir=out_dir,
        on_progress=on_prog,
        on_complete=on_comp,
    )

    assert isinstance(thread, threading.Thread)
    thread.join(timeout=5.0)

    assert os.path.exists(os.path.join(out_dir, "rr_1_5.png"))
    assert os.path.exists(os.path.join(out_dir, "rr_2_9.png"))
    assert len(progress_events) == 2
    assert len(complete_events) == 1
    assert len(complete_events[0]) == 2


def test_render_query_results_rich_clean(capsys):
    """Verify render_query_results_rich displays top matching pages cleanly without background banner."""
    from bbq.src.terminal import render_query_results_rich

    results = [
        {"file_path": "p1.pdf", "page_number": 10},
        {"file_path": "p2.pdf", "page_number": 20},
    ]

    render_query_results_rich(results)
    out = capsys.readouterr().out
    assert "Top 2 Matching PDF Pages" in out
    assert "Saving in bg" not in out
    assert "p1.pdf" in out
    assert "p2.pdf" in out
