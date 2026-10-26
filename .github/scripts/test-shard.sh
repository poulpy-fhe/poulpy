#!/usr/bin/env bash
# Pack the test executables of a `cargo test --no-run` build, or run one shard of a pack.
set -euo pipefail

stage=${1:?expected pack or run}
pack=${2:?expected a pack directory}

case "$stage" in
  pack)
    # Reads `cargo test --no-run --message-format=json` on stdin.
    mkdir -p "$pack"
    jq -r 'select(.reason == "compiler-artifact" and .profile.test and .executable != null)
      | "\(.executable) \(.manifest_path)"' |
      while read -r executable manifest; do
        cp "$executable" "$pack/"
        echo "$(basename "$executable") $(realpath --relative-to=. "$(dirname "$manifest")")" >> "$pack/manifest"
      done
    ;;
  run)
    shard=${3:?expected a 1-based shard index}
    shards=${4:?expected a shard count}
    pack=$(realpath "$pack")
    # Every shard sees the same sorted list, so round-robin selection covers each test once.
    while read -r executable package; do
      # Listed outside the process substitution so a binary that cannot start fails the shard.
      listed=$("$pack/$executable" --list --format terse < /dev/null)
      mapfile -t tests < <(sed -n 's/: test$//p' <<< "$listed" | awk -v k="$shard" -v n="$shards" '(NR - 1) % n == k - 1')
      # An empty filter list would run the whole binary.
      [[ ${#tests[@]} -gt 0 ]] || continue
      echo "::group::$executable (${#tests[@]} tests)"
      (cd "$package" && "$pack/$executable" --exact "${tests[@]}" -Z unstable-options --report-time < /dev/null)
      echo "::endgroup::"
    done < "$pack/manifest"
    ;;
  *) echo "Invalid stage: $stage" >&2; exit 2 ;;
esac
