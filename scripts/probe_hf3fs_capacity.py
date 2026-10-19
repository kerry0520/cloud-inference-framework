"""CPU probe of original HF3FS state classes; not a distributed IO test."""
import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import threading


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False, parents=True)
    data = args.source.read_bytes()
    tree = ast.parse(data)
    names = {'RankFileMetadata', 'KeyMetadata', 'GlobalMetadataState'}
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    assert {n.name for n in nodes} == names
    scope = dict(dataclass=dataclass, threading=threading, logger=logging.getLogger('probe'))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(args.source), 'exec'), scope)
    state = scope['GlobalMetadataState']()
    state.initialize_rank(0, 2)
    allocations = []
    for key in ('a', 'b', 'c'):
        mapping = state.allocate_pages_for_keys(0, [(key, '')])
        page = mapping[key]
        allocations.append([key, page])
        if page >= 0:
            state.confirm_write_for_keys(0, [(key, page)])
    assert allocations == [['a', 0], ['b', 1], ['c', -1]]
    assert state.batch_key_exists(['a', 'b', 'c']) == [True, True, False]
    # Existing complete keys remain reusable even when the free pool is empty.
    reuse = state.allocate_pages_for_keys(0, [('a', '')])
    assert reuse == {'a': 0}
    # A failed write explicitly releasing a page does restore capacity.
    released = scope['GlobalMetadataState']()
    released.initialize_rank(0, 1)
    page = released.allocate_pages_for_keys(0, [('failed', '')])['failed']
    released.confirm_write_for_keys(0, [], [page])
    after_release = released.allocate_pages_for_keys(0, [('new', '')])
    assert after_release == {'new': 0}
    # Less free space than a whole batch: even the available page is not assigned.
    partial = scope['GlobalMetadataState']()
    partial.initialize_rank(0, 1)
    oversized = partial.allocate_pages_for_keys(0, [('a', ''), ('b', '')])
    assert oversized == {'a': -1, 'b': -1}
    assert partial.rank_metadata[0].get_free_page_count() == 1
    result = dict(source=str(args.source), sha256=hashlib.sha256(data).hexdigest(),
                  extraction='Unmodified AST of three original classes; stdlib dependencies only',
                  allocations=allocations, reuse_when_full=reuse,
                  failed_write_release=after_release, oversized_batch=oversized,
                  exitcode=0, limitations='Single-rank CPU metadata only; no HTTP, GPU, HF3FS IO, concurrent readers or fix validation')
    (args.out / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
