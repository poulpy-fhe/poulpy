#!/usr/bin/env python3
"""Check the explicit core method inventory against rustc metadata and test output.

The pinned nightly's rustdoc JSON is the source of truth for trait methods, not
regexes over declarations. Execution validation accepts successful libtest output,
not `--list`, so an ignored, filtered, unregistered or failing test is insufficient.
"""

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "poulpy-core/docs/core-contracts.json"
EXTERNAL_TRAITS = {"BSGSOps", "DiagonalProd", "GLWEKeyswitchInternal", "GetAutomorphismKey", "GetTensorKey", "GLWESecretTensorFactory"}


def selected_trait(item):
    if item["crate_id"] != 0 or "trait" not in item["inner"]:
        return False
    source = (item.get("span") or {}).get("filename", "")
    name = item["name"]
    return (
        "/src/oep/" in source
        or "/src/reference/" in source
        and (name.endswith("Reference") or name in EXTERNAL_TRAITS)
        or "/src/layouts/" in source
        and (name.endswith("PreparedFactory") or name.endswith("Decompress") or name in EXTERNAL_TRAITS)
    )


def collect(metadata):
    index = metadata["index"]
    methods = {}
    symbols = set()
    for item in index.values():
        if item["crate_id"] != 0:
            continue
        source = (item.get("span") or {}).get("filename", "")
        if "function" in item["inner"]:
            symbols.add(source + "#" + item["name"])
        if not selected_trait(item):
            continue
        for child_id in item["inner"]["trait"]["items"]:
            child = index[str(child_id)]
            if "function" in child["inner"]:
                key = item["name"] + "::" + child["name"]
                if key in methods:
                    raise ValueError("ambiguous method identity: " + key)
                methods[key] = {"source": source, "trait": item, "method": child}
    return methods, symbols


def registrations(package):
    """Read the concrete shared-suite invocations; rustc still compiles/runs them."""
    source = (ROOT / package / "src/tests.rs").read_text()
    result = {}
    for start in re.finditer(r"core_parity_test_suite!\s*\{", source):
        begin = start.end()
        depth, end = 1, begin
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        block = source[begin:end - 1]
        module = re.search(r"\bmod\s+(\w+)", block).group(1)
        if "rank1" in module or "_fused" in module:  # Additional restricted-envelope regression suite.
            continue
        backend = re.search(r"backend_test\s*=\s*([\w:]+)", block).group(1).split("::")[-1]
        oracle = re.search(r"backend_ref\s*=\s*([\w:]+)", block).group(1)
        tests = dict(re.findall(r"(\w+)\s*=>\s*[\w:$]+::(test_\w+)", block))
        if backend in result:
            raise ValueError("multiple full contract suites for " + backend)
        result[backend] = ("tests::" + module + "::", tests, oracle)
    return result


def encryption_registrations(package):
    source = (ROOT / package / "src/tests.rs").read_text()
    modules = {}
    for match in re.finditer(r"core_encryption_parity_test_suite!\(mod\s+(\w+),\s*backend\s*=\s*([\w:]+)\)", source):
        modules[match[2].split("::")[-1]] = "tests::" + match[1] + "::"
    macro = (ROOT / "poulpy-cpu-ref/src/test_suite/controlled_sampling.rs").read_text()
    macro_start = macro.index("macro_rules! core_encryption_parity_test_suite")
    begin = macro.index("{", macro_start) + 1
    depth, end = 1, begin
    while depth:
        depth += (macro[end] == "{") - (macro[end] == "}")
        end += 1
    macro = macro[begin:end - 1]
    tests = {}
    for match in re.finditer(r"#\[test\]\s*fn\s+(\w+)\(\)\s*\{", macro):
        depth, end = 1, match.end()
        while depth:
            depth += (macro[end] == "{") - (macro[end] == "}")
            end += 1
        calls = re.findall(r"\b(test_\w+)\(", macro[match.end():end])
        if len(calls) != 1:
            raise ValueError("randomized test must call one documented shared suite: " + match[1])
        tests[match[1]] = calls[0]
    return modules, tests


def check(manifest, metadata, runs):
    errors = []
    actual, symbols = collect(metadata)
    bodies = {
        (item.get("span") or {}).get("filename", "") + "#" + item["name"]
        for item in metadata["index"].values()
        if item["crate_id"] == 0 and "function" in item["inner"]
        and item["inner"]["function"]["has_body"]
    }
    recorded = manifest["methods"]
    for name in sorted(actual.keys() - recorded.keys()):
        errors.append("missing method: " + name)
    for name in sorted(recorded.keys() - actual.keys()):
        errors.append("stale method: " + name)
    for name in sorted(actual.keys() & recorded.keys()):
        row = recorded[name]
        if row["source"] != actual[name]["source"]:
            errors.append("incorrect source: " + name)
        if not row.get("dispatch") or not row.get("reference"):
            errors.append("missing dispatch/reference mapping: " + name)
        if not row.get("tests"):
            errors.append("missing executable contract test: " + name)
        for dispatch in row.get("dispatch", []):
            if dispatch not in symbols:
                errors.append(f"missing public/internal dispatch {dispatch}: {name}")
        for test in row.get("tests", []):
            if test not in manifest["tests"]:
                errors.append(f"unknown test {test}: {name}")
        for reference in row.get("reference", []):
            if reference.startswith("exception:"):
                if reference[10:] not in manifest["exceptions"]:
                    errors.append(f"unknown reference exception {reference}: {name}")
            elif reference not in bodies:
                errors.append(f"missing reference body {reference}: {name}")

    for test, entry in manifest["tests"].items():
        if entry["function"] not in bodies:
            errors.append(f"test function absent from compiler metadata: {test}")
        if not entry.get("oracle"):
            errors.append(f"missing test oracle: {test}")
        if not entry.get("registration"):
            errors.append(f"missing libtest registration name: {test}")

    registered = {}
    randomized = {}
    random_tests = {}
    for package in {entry["package"] for entry in manifest["backends"].values()}:
        registered.update(registrations(package))
        modules, random_tests = encryption_registrations(package)
        randomized.update(modules)
    for backend in registered.keys() - manifest["backends"].keys():
        errors.append("backend missing from contract inventory: " + backend)
    for backend, entry in manifest["backends"].items():
        if backend not in registered:
            errors.append("backend contract suite not registered: " + backend)
            continue
        prefix, tests, oracle = registered[backend]
        oracle_crate = "crate::" if entry["package"] == "poulpy-cpu-ref" else "poulpy_cpu_ref::"
        expected_oracle = oracle_crate + entry["reference_backend"]
        if entry["reference_backend"] not in {"FFT64Ref", "NTT4x30Ref"} or oracle != expected_oracle:
            errors.append("incorrect portable reference backend: " + backend)
        if prefix != entry["test_prefix"]:
            errors.append("incorrect test prefix: " + backend)
        for test, contract in manifest["tests"].items():
            if backend in contract.get("unsupported", {}):
                continue
            function = contract["function"].split("#")[-1]
            available = tests
            if contract.get("suite") == "encryption":
                available = random_tests
                if randomized.get(backend) != entry.get("encryption_prefix"):
                    errors.append("missing randomized suite registration: " + backend)
            if available.get(contract["registration"]) != function:
                errors.append(f"missing supported registration: {backend}/{test}")

    for backend, output in runs:
        if backend not in manifest["backends"]:
            errors.append("unknown backend: " + backend)
            continue
        contract = manifest["backends"][backend]
        successful = set(re.findall(r"^test (\S+) \.\.\. ok$", Path(output).read_text(), re.MULTILINE))
        prefix = contract["test_prefix"]
        for test, entry in manifest["tests"].items():
            if backend in entry.get("unsupported", {}):
                if not entry["unsupported"][backend]:
                    errors.append(f"missing unsupported reason: {test}/{backend}")
                continue
            suite_prefix = contract.get("encryption_prefix") if entry.get("suite") == "encryption" else prefix
            expected = (suite_prefix or "MISSING::") + entry["registration"]
            if expected not in successful:
                errors.append(f"contract test did not pass: {backend}/{expected}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rustdoc-json", type=Path, default=ROOT / "target/doc/poulpy_core.json")
    parser.add_argument("--run", nargs=2, action="append", default=[], metavar=("BACKEND", "LOG"))
    parser.add_argument("--run-package", nargs=2, action="append", default=[], metavar=("PACKAGE", "LOG"))
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    metadata = json.loads(args.rustdoc_json.read_text())
    manifest = json.loads(args.manifest.read_text())
    for package, output in args.run_package:
        backends = [name for name, entry in manifest["backends"].items() if entry["package"] == package]
        if not backends:
            parser.error("package has no supported backends: " + package)
        args.run.extend((backend, output) for backend in backends)
    errors = check(manifest, metadata, args.run)
    if errors:
        print("Core contract inventory failed:", file=sys.stderr)
        print("\n".join("- " + error for error in errors), file=sys.stderr)
        return 1
    print(f"Core contract inventory: {len(manifest['methods'])} methods, "
          f"{len(manifest['tests'])} contract tests; {len(args.run)} backend execution logs verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
