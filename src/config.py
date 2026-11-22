"""Fixed training recipe for the complete manuscript model."""
DATASETS = ['1-circAtlas', '2-CircR2Disease', '3-circRNADisease']
DATA_FILES = ['circRNA_RNAFM_PCA_64.csv', 'disease_features_64d.csv',
              'interaction.csv', 'sample_pairs_easy_far_dual.csv']
BASE = dict(dim=64, device='cpu', lr=0.001, min_lr=0.00002, epochs=100,
            patience=20, batch_size=512, validation_only=True,
            selection_metric='logloss', ema_decay=0.0)
GRID = [dict(hidden=h, dropout=d, k=k, temperature=t, weight_decay=w,
             edge_dropout=0.0, input_dropout=p)
        for p in [0.0, 0.15, 0.30]
        for h, d, k, t, w in [(32, 0.1, 10, 0.15, 0.001),
                              (64, 0.3, 20, 0.30, 0.003)]]
