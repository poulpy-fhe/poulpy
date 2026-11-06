#!/usr/bin/env python3
"""Checks the code names of a CHANGELOG release section against the code.

Run it from the repository root before tagging, with the version of the section and
the tag of the previous release:

    python3 tools/changelog_check.py 0.10.0 v0.9.0

Every Rust identifier in a backticked span of the section must exist in the Rust
sources of `HEAD` (a new name) or of the previous tag (an old name). Braces `{a, b}`
and optional brackets `[x]` or `[x, y]` expand, a span holding a placeholder such as
`impl_<op>_reference_full!` or `*_tmp_bytes` or a formula is skipped, and a name that
prefixes an existing one passes. Plain words are not checked: a name needs an
underscore or an inner capital. The old names of a crate moved in from another
repository show up and are checked by hand.
"""

import bisect
import re
import subprocess
import sys

PLACEHOLDER = re.compile(r"_<|>[A-Za-z_]|^<|^_|\*|[^\x00-\x7f]")
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def section(version):
    with open("CHANGELOG.md") as f:
        lines = f.read().split("\n")
    start = next((i for i, l in enumerate(lines) if l.startswith(f"## [{version}]")), None)
    if start is None:
        sys.exit(f"CHANGELOG.md has no section {version}")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## [")), len(lines))
    return [(i + 1, lines[i]) for i in range(start, end)]


def identifiers(rev):
    out = subprocess.run(
        ["git", "grep", "-ohE", TOKEN.pattern, rev, "--", "*.rs"],
        capture_output=True, text=True, check=True,
    ).stdout
    return set(out.split())


def expand(span):
    m = re.search(r"\{([^{}]*)\}", span)
    if m:
        return [x for alt in m.group(1).split(",") for x in expand(span[: m.start()] + alt.strip() + span[m.end():])]
    m = re.search(r"\[([^\[\]]*)\]", span)
    if m:
        body = m.group(1)
        alts = [""] + ([body] if body.startswith(",") else [alt.strip() for alt in body.split(",")])
        return [x for alt in alts for x in expand(span[: m.start()] + alt + span[m.end():])]
    return [span]


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    version, previous = sys.argv[1:]
    known = sorted(identifiers("HEAD") | identifiers(previous))
    missing = 0
    for lineno, line in section(version):
        for span in re.findall(r"`([^`]+)`", line):
            if PLACEHOLDER.search(span):
                continue
            for name in {t for s in expand(span) for t in TOKEN.findall(s)}:
                if len(name) < 5 or not ("_" in name or re.search(r"[a-z0-9][A-Z]", name)):
                    continue
                i = bisect.bisect_left(known, name)
                if i == len(known) or not known[i].startswith(name):
                    print(f"CHANGELOG.md:{lineno}: `{name}` is in neither HEAD nor {previous}")
                    missing += 1
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
