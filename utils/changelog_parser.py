import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file', required=True)
parser.add_argument('--runner-config', required=True)
args = parser.parse_args()

pattern = re.compile(r'^([^|]+)\|([^|]+)$')

with open(args.file) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        match = pattern.match(line)
        if not match:
            raise ValueError(f"Invalid line: {line}")
        name, comment = match.groups()
        print((name.strip(), comment.strip()))