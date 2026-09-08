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
        default="maxsimd",
        choices=["maxsimd", "vidore"],
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
        type=lambda x: x.lower() not in ("false", "0", "no"),
        default=True,
        help="Generate report (default: True)",
    )

    return parser


def main():
    p = bm_parser()
    args = p.parse_args()

    print_bbq(name=args.suite)
    configure_global_threads(args.jobs)

    if args.suite == "maxsimd":
        from benchmarks.maxsimd.run import run

        run(
            q_len=args.q_len,
            tokens_per_doc=args.dim,
            docs=args.docs,
            dim=args.dim,
            seed=args.seed,
            jobs=args.jobs,
            report=args.report
        )
    # elif args.suite == "vidore":
    #     from benchmarks.vidore.run import run
    #
    #     run(
    #         q_len=args.q_len,
    #         tokens_per_doc=args.dim,
    #         num_docs=args.docs,
    #         dim=args.dim,
    #         seed=args.seed,
    #         jobs=args.jobs,

    else:
        raise ValueError(f"Unknown benchmark suite: {args.suite}")
