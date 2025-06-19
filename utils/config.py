import yaml, argparse, pathlib, copy

_DEFAULT = yaml.safe_load((pathlib.Path(__file__).parent.parent /
                           "config/defaults.yaml").read_text())

def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=pathlib.Path,
                   help="YAML file overriding defaults")
    args, unknown = p.parse_known_args()
    cfg = copy.deepcopy(_DEFAULT)
    if args.config:
        cfg = _merge(cfg, yaml.safe_load(args.config.read_text()))
    return cfg, unknown

def _merge(base, new):
    for k,v in new.items():
        if isinstance(v, dict):
            base[k] = _merge(base.get(k, {}), v)
        else:
            base[k] = v
    return base