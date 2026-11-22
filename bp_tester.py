"""Predict saved-fold test pairs or a query CSV with one complete BP checkpoint."""
import argparse
from pathlib import Path

from src.config import DATASETS
from src.data import read_queries
from src.metrics import report_metrics
from src.prediction import load_checkpoint, predict_scores, test_predictions
from src.utils import ROOT, single_thread, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--fold', type=int, choices=range(5), default=0)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data')
    parser.add_argument('--checkpoint-dir', type=Path,
                        help='A model.pt/config.json/split.npz/checkpoint.json directory; overrides --fold.')
    parser.add_argument('--pairs', type=Path, help='Optional CSV with circRNA,disease[,label] header.')
    parser.add_argument('--output', type=Path, required=True, help='New output CSV path.')
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() or output.with_suffix('.metrics.json').exists():
        parser.error('Output CSV or metrics JSON already exists; use a new path.')
    controller = single_thread()
    folder = args.checkpoint_dir or ROOT/'weights'/args.dataset/f'fold{args.fold}'
    model,encoded,data,split,metadata = load_checkpoint(args.data_dir,folder,args.dataset)
    if args.pairs:
        frame,pairs = read_queries(args.pairs,data)
        frame['score'] = predict_scores(model,encoded,pairs).astype('float64')
    else:
        frame = test_predictions(model,encoded,data,split)
    frame['predicted_label'] = (frame.score >= 0.5).astype(int)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output,index=False)
    if 'label' in frame:
        values = report_metrics(frame.label,frame.score)
        values.update(dataset=args.dataset,fold=metadata['fold'],prediction_model_count=1,
                      checkpoint_sha256=metadata['files_sha256']['model.pt'])
        write_json(output.with_suffix('.metrics.json'),values)
    print(f'Saved {len(frame)} predictions from one raw checkpoint to {output}')


if __name__ == '__main__':
    main()
