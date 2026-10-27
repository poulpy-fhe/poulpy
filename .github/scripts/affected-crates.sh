#!/usr/bin/env bash
# Print, as a JSON array, the workspace crates changed since a base revision plus every crate depending on
# them. Without a base revision, or when a changed file lies outside every crate, print all workspace crates.
set -euo pipefail

base=${1:-}
metadata=$(cargo metadata --format-version 1 --no-deps)
changed=null
if [[ -n "$base" ]]; then
  changed=$(git diff --name-only "$base" HEAD | jq -Rcs 'split("\n") | map(select(. != ""))')
fi

jq -c --argjson changed "$changed" '
  # The deepest crate directory holding the file, or null outside every crate.
  def owner($crates): . as $file
    | [$crates[] | select(.dir as $dir | $file | startswith($dir + "/"))] | max_by(.dir | length) | .name?;
  def dependents($crates): . as $set
    | ($set + [$crates[] | select(any(.deps[]; IN($set[]))) | .name] | unique) as $next
    | if $next == $set then $set else $next | dependents($crates) end;

  .workspace_root as $root
  | [.packages[] | {name, dir: (.manifest_path | ltrimstr($root + "/") | rtrimstr("/Cargo.toml")),
      deps: [.dependencies[].name]}] as $crates
  | [$crates[].name] as $all
  | if $changed == null then $all
    else [$changed[] | owner($crates)] as $owners
      | if any($owners[]; . == null) then $all else $owners | unique | dependents($crates) end
    end
' <<< "$metadata"
