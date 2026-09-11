import argparse
from benchmarks.ui import print_bbq
from benchmarks.common import configure_global_threads


def bm_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Suite")

    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Number of threads for maxsimd (default: -1 for all cores, 1 for sequential)",
    )

    parser.add_argument(
        "--docs",
        type=int,
        nargs="+",
        default=[
            20,
            50,
            100,
            250,
            500,
            1000,
            2000,
            3000,
            4000,
            5000,
            6000,
            7000,
            8000,
            9000,
            10000,
        ],
        help="List of document counts to benchmark (e.g. --docs 20 50 100)",
    )

    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        choices=["maxsimd", "vidore", "all"],
        help="Benchmark to run (e.g. --suite maxsimd) options: maxsimd, vidore",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed to use for random number generation (default: 42)",
    )

    parser.add_argument(
        "--dim",
        type=int,
        default=128,
        help="Dimensionality of query and document vectors (default: 128)",
    )

    parser.add_argument(
        "--q_len",
        type=int,
        default=32,
        help="Length of query vector (default: 32)",
    )

    parser.add_argument(
        "--report",
        nargs="?",
        const=True,
        default=True,
        type=lambda x: x.lower() not in ("false", "0", "no"),
        help="Generate report (default: True)",
    )

    parser.add_argument(
        "--dataset",
        "--datasets",
        nargs="+",
        default=None,
        help="Specific ViDoRe dataset(s) to benchmark (default: all)",
    )

    parser.add_argument(
        "--force",
        nargs="?",
        const=True,
        default=False,
        type=lambda x: x.lower() not in ("false", "0", "no"),
        help="Force re-download and re-embedding for ViDoRe (default: False)",
    )

    parser.add_argument(
        "--data-dir",
        "--data_dir",
        type=str,
        default="data/vidore",
        help="Directory for ViDoRe datasets (default: data/vidore)",
    )

    parser.add_argument(
        "--emb-dir",
        "--emb_dir",
        type=str,
        default="data/embeddings",
        help="Directory for ViDoRe embeddings (default: data/embeddings)",
    )

    parser.add_argument(
        "--asset",
        "--assets",
        "--asset-dir",
        "--asset_dir",
        dest="asset_dir",
        nargs="?",
        const="assets",
        default=None,
        help="Save b1/b2 benchmark images to asset directory (default: None, only saves in bench/ folder. Pass --asset to save to assets/)",
    )

    return parser


def main():
    p = bm_parser()
    args = p.parse_args()

    configure_global_threads(args.jobs)

    if args.suite == "all":
        print_bbq(name=args.suite)
        from benchmarks.maxsimd.run import run as run_maxsimd

        run_maxsimd(
            q_len=args.q_len,
            tokens_per_doc=args.dim,
            docs=args.docs,
            dim=args.dim,
            seed=args.seed,
            jobs=args.jobs,
            report=args.report,
            asset_dir=args.asset_dir,
        )

        from benchmarks.vidore.run import run as run_vidore

        run_vidore(
            datasets=args.dataset,
            force=args.force,
            report=args.report,
            data_dir=args.data_dir,
            emb_dir=args.emb_dir,
            asset_dir=args.asset_dir,
        )

    elif args.suite == "maxsimd":
        print_bbq(name=args.suite)
        from benchmarks.maxsimd.run import run

        run(
            q_len=args.q_len,
            tokens_per_doc=args.dim,
            docs=args.docs,
            dim=args.dim,
            seed=args.seed,
            jobs=args.jobs,
            report=args.report,
            asset_dir=args.asset_dir,
        )
    elif args.suite == "vidore":
        print_bbq(name=args.suite)
        from benchmarks.vidore.run import run

        run(
            datasets=args.dataset,
            force=args.force,
            report=args.report,
            data_dir=args.data_dir,
            emb_dir=args.emb_dir,
            asset_dir=args.asset_dir,
        )
    else:
        raise ValueError(f"Unknown benchmark suite: {args.suite}")
