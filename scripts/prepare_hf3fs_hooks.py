"""Select original Python-applicable hook repositories, without changing vLLM."""
import argparse
from pathlib import Path
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--repo', required=True, type=Path)
parser.add_argument('--out', required=True, type=Path)
args = parser.parse_args()
config = yaml.safe_load((args.repo / '.pre-commit-config.yaml').read_text())
keep = {'local', 'https://github.com/astral-sh/ruff-pre-commit', 'https://github.com/crate-ci/typos'}
config['repos'] = [repo for repo in config['repos'] if repo['repo'] in keep]
with args.out.open('x', encoding='utf-8') as stream:
    yaml.safe_dump(config, stream, sort_keys=False)
