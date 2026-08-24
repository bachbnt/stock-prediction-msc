"""Versioned model artifacts tied to an exact training configuration."""
import hashlib
import json
import math
import os
from datetime import datetime, timezone

import joblib


ARTIFACT_VERSION = 1


def build_artifact_config(symbol: str, date_milestone: str, window: int,
                          n_steps: int, feature_cols,
                          sentiment_model: str = None,
                          seed: int = None,
                          batch_size: int = None) -> dict:
    """Return the stable configuration used to identify compatible artifacts."""
    return {
        'artifact_version': ARTIFACT_VERSION,
        'symbol': symbol.upper(),
        'date_milestone': str(date_milestone),
        'window': int(window),
        'n_steps': int(n_steps),
        'feature_cols': list(feature_cols),
        'sentiment_model': (os.path.normpath(sentiment_model)
                            if sentiment_model else None),
        'seed': seed,
        'batch_size': batch_size,
    }


def artifact_directory(config: dict, model_root: str = 'models') -> str:
    """Build a readable path with a hash that covers the full configuration."""
    serialized = json.dumps(config, ensure_ascii=True, sort_keys=True,
                            separators=(',', ':'))
    config_hash = hashlib.sha256(serialized.encode('utf-8')).hexdigest()[:10]
    name = (f"w{config['window']}_s{config['n_steps']}_"
            f"f{len(config['feature_cols'])}_{config_hash}")
    return os.path.join(model_root, config['symbol'], name)


def load_artifact_bundle(artifact_dir: str, expected_config: dict):
    """Load metadata and scaler when they exactly match the active config."""
    metadata_path = os.path.join(artifact_dir, 'metadata.json')
    scaler_path = os.path.join(artifact_dir, 'scaler.joblib')
    if not os.path.exists(metadata_path) or not os.path.exists(scaler_path):
        return None, None, 'metadata.json or scaler.joblib is missing'

    try:
        with open(metadata_path, encoding='utf-8') as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return None, None, f'cannot read metadata: {exc}'

    if metadata.get('config') != expected_config:
        return None, None, 'saved configuration does not match the active run'

    try:
        scaler = joblib.load(scaler_path)
    except Exception as exc:
        return None, None, f'cannot load scaler: {exc}'

    expected_features = len(expected_config['feature_cols'])
    if getattr(scaler, 'n_features_in_', None) != expected_features:
        return None, None, 'saved scaler has an incompatible feature count'

    val_losses = metadata.get('val_losses')
    if not isinstance(val_losses, dict):
        return None, None, 'validation losses are missing from metadata'
    try:
        valid_losses = all(math.isfinite(float(loss)) and float(loss) >= 0
                           for loss in val_losses.values())
    except (TypeError, ValueError):
        valid_losses = False
    if not valid_losses:
        return None, None, 'metadata contains invalid validation losses'

    return metadata, scaler, None


def save_artifact_bundle(artifact_dir: str, config: dict, scaler,
                         val_losses: dict, training_data: dict = None) -> None:
    """Persist the shared scaler and model validation losses atomically."""
    os.makedirs(artifact_dir, exist_ok=True)
    scaler_path = os.path.join(artifact_dir, 'scaler.joblib')
    scaler_tmp = f'{scaler_path}.tmp'
    joblib.dump(scaler, scaler_tmp)
    os.replace(scaler_tmp, scaler_path)

    metadata = {
        'config': config,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'training_data': training_data or {},
        'val_losses': {name: float(loss) for name, loss in val_losses.items()},
    }
    metadata_path = os.path.join(artifact_dir, 'metadata.json')
    metadata_tmp = f'{metadata_path}.tmp'
    with open(metadata_tmp, 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(metadata_tmp, metadata_path)
