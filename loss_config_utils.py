import hashlib

import yaml


SUPPORTED_LOSS_TYPES = {'l1', 'l2', 'pseudo_huber', 'inception', 'lpips', 'dinov2', 'lpl', 'multiscale', 'gradient', 'ms_ssim', 'structure'}
SUPPORTED_LOSS_TARGETS = {'state', 'clean'}
SUPPORTED_LOSS_GATES = {'state', 'clean', 'struct'}
DEFAULT_SNR_GATE_CONFIG = {
    'tau': 0.0,
    'kappa': 1.0,
    'p': 1.0,
    'q': 2.0,
    'w_min': 0.05,
    'sigma_data': 0.5,
}
DEFAULT_ROLLOUT_CONFIG = {
    'enabled': False,
}
DEFAULT_STATE_NORMALIZATION_CONFIG = {
    'enabled': False,
    'sigma_data': 0.5,
}


def load_loss_config(loss_config_path):
    with open(loss_config_path, 'r') as f:
        loss_config_text = f.read()
    loss_config = yaml.safe_load(loss_config_text)
    if not isinstance(loss_config, dict):
        raise ValueError(f'Loss config must be a YAML mapping: {loss_config_path}')
    return normalize_loss_config(loss_config), loss_config_text


def load_loss_id(loss_config_path):
    loss_config, _ = load_loss_config(loss_config_path)
    return loss_config['loss_id']


def normalize_loss_config(loss_config):
    loss_tag = loss_config.get('loss_tag')
    if not isinstance(loss_tag, str) or len(loss_tag) == 0:
        raise ValueError('loss_config must define a non-empty string "loss_tag"')

    middle = normalize_loss_items(loss_config.get('middle'), 'middle')
    endpoint = normalize_loss_items(loss_config.get('endpoint'), 'endpoint')
    normalized = {
        'loss_tag': loss_tag,
        'middle': middle,
        'endpoint': endpoint,
    }
    if loss_config.get('snr_gate') is not None or loss_items_use_gate(middle + endpoint):
        normalized['snr_gate'] = normalize_snr_gate_config(loss_config.get('snr_gate'))
    if loss_config.get('rollout') is not None:
        normalized['rollout'] = normalize_rollout_config(loss_config.get('rollout'))
    if loss_config.get('state_normalization') is not None:
        normalized['state_normalization'] = normalize_state_normalization_config(loss_config.get('state_normalization'))
    normalized['loss_hash'] = compute_loss_hash(normalized)
    normalized['loss_id'] = f'{loss_tag}__{normalized["loss_hash"]}'
    return normalized


def normalize_state_normalization_config(state_normalization_config):
    if isinstance(state_normalization_config, bool):
        state_normalization_config = {'enabled': state_normalization_config}
    if state_normalization_config is None:
        return DEFAULT_STATE_NORMALIZATION_CONFIG.copy()
    if not isinstance(state_normalization_config, dict):
        raise ValueError('loss_config["state_normalization"] must be a mapping or bool')

    normalized = DEFAULT_STATE_NORMALIZATION_CONFIG.copy()
    if 'enabled' in state_normalization_config:
        normalized['enabled'] = bool(state_normalization_config['enabled'])
    if 'sigma_data' in state_normalization_config:
        normalized['sigma_data'] = float(state_normalization_config['sigma_data'])
    if normalized['sigma_data'] <= 0:
        raise ValueError('loss_config["state_normalization"]["sigma_data"] must be positive')
    return normalized


def normalize_rollout_config(rollout_config):
    if isinstance(rollout_config, bool):
        return {'enabled': rollout_config}
    if rollout_config is None:
        return DEFAULT_ROLLOUT_CONFIG.copy()
    if not isinstance(rollout_config, dict):
        raise ValueError('loss_config["rollout"] must be a mapping or bool')

    normalized = DEFAULT_ROLLOUT_CONFIG.copy()
    if 'enabled' in rollout_config:
        normalized['enabled'] = bool(rollout_config['enabled'])
    return normalized


def normalize_snr_gate_config(gate_config):
    if gate_config is None:
        gate_config = {}
    if not isinstance(gate_config, dict):
        raise ValueError('loss_config["snr_gate"] must be a mapping')

    normalized = DEFAULT_SNR_GATE_CONFIG.copy()
    for key in normalized:
        if key in gate_config:
            normalized[key] = float(gate_config[key])
    if normalized['kappa'] <= 0:
        raise ValueError('loss_config["snr_gate"]["kappa"] must be positive')
    if normalized['sigma_data'] <= 0:
        raise ValueError('loss_config["snr_gate"]["sigma_data"] must be positive')
    if normalized['w_min'] < 0:
        raise ValueError('loss_config["snr_gate"]["w_min"] must be non-negative')
    return normalized


def normalize_loss_items(items, stage_name):
    if not isinstance(items, list):
        raise ValueError(f'loss_config["{stage_name}"] must be a list')

    normalized = []
    for idx, item in enumerate(items):
        if item is None:
            continue
        if not isinstance(item, dict):
            raise ValueError(f'loss_config["{stage_name}"][{idx}] must be a mapping')

        loss_type = item.get('type')
        target = item.get('target')
        weight = float(item.get('weight', 0.0))
        if weight == 0.0:
            continue
        if loss_type not in SUPPORTED_LOSS_TYPES:
            raise ValueError(f'Unsupported loss type in {stage_name}[{idx}]: {loss_type}')
        if target not in SUPPORTED_LOSS_TARGETS:
            raise ValueError(f'Unsupported loss target in {stage_name}[{idx}]: {target}')

        normalized_item = {'type': loss_type, 'target': target, 'weight': weight}
        gate = item.get('gate')
        if gate == 'structure':
            gate = 'struct'
        if gate is not None:
            if gate not in SUPPORTED_LOSS_GATES:
                raise ValueError(f'Unsupported loss gate in {stage_name}[{idx}]: {gate}')
            normalized_item['gate'] = gate
        if 'delta' in item:
            normalized_item['delta'] = float(item['delta'])
        elif loss_type == 'pseudo_huber':
            raise ValueError(f'loss_config["{stage_name}"][{idx}] uses pseudo_huber and must define "delta"')
        normalized.append(normalized_item)

    if len(normalized) == 0:
        raise ValueError(f'loss_config["{stage_name}"] must contain at least one active item with weight != 0')
    return normalized


def loss_items_use_gate(loss_items):
    return any('gate' in item for item in loss_items)


def compute_loss_hash(normalized_loss_config, hash_len=10):
    hash_payload = {
        'middle': normalized_loss_config['middle'],
        'endpoint': normalized_loss_config['endpoint'],
    }
    if 'snr_gate' in normalized_loss_config:
        hash_payload['snr_gate'] = normalized_loss_config['snr_gate']
    if 'rollout' in normalized_loss_config:
        hash_payload['rollout'] = normalized_loss_config['rollout']
    if 'state_normalization' in normalized_loss_config:
        hash_payload['state_normalization'] = normalized_loss_config['state_normalization']
    canonical_text = yaml.safe_dump(hash_payload, sort_keys=True)
    return hashlib.sha256(canonical_text.encode('utf-8')).hexdigest()[:hash_len]
