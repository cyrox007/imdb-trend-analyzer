def count_lines_gz(filepath):
    import gzip
    with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
        return sum(1 for _ in f)