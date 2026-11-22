"""Train the complete BalancedPathFusion model with validation-only selection."""
import argparse
from pathlib import Path

from src.config import DATASETS
from src.study import evaluate, train
from src.utils import ROOT, single_thread


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['train','evaluate','all'], default='all')
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data')
    parser.add_argument('--output', type=Path, default=ROOT/'results/training')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--folds', nargs='+', type=int, choices=range(5), default=list(range(5)))
    parser.add_argument('--candidates', nargs='+', type=int, choices=range(6), default=list(range(6)))
    parser.add_argument('--workers', type=int, choices=range(1,9), default=6)
    args = parser.parse_args()
    for name in ['datasets','folds','candidates']:
        values = getattr(args,name)
        if len(values) != len(set(values)):
            parser.error(f'--{name} contains duplicates.')
    controller = single_thread()
    data_dir, output = args.data_dir.expanduser().resolve(), args.output.expanduser().resolve()
    if args.phase in ['train','all']:
        train(data_dir,output,args.datasets,args.folds,args.candidates,args.workers)
    if args.phase in ['evaluate','all']:
        evaluate(data_dir,output)


if __name__ == '__main__':
    main()
