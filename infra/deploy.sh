#!/usr/bin/env bash
# Runs on the server: infra/deploy.sh <image-tag>. Also the manual rollback, given an older commit SHA.
set -euo pipefail

cd "$(dirname "$0")"
tag="${1:?usage: deploy.sh <image-tag>}"
tag_file=../.deployed-tag
previous="$(cat "$tag_file" 2>/dev/null || true)"

compose() {
  IMAGE_TAG="$1" docker compose --env-file ../.env \
    -f docker-compose.yml -f docker-compose.prod.yml "${@:2}"
}

start() {
  compose "$1" pull --quiet
  compose "$1" up --detach --no-build --wait --wait-timeout 300 --remove-orphans
}

if ! start "$tag"; then
  echo "deploy of $tag failed" >&2
  compose "$tag" ps --all >&2 || true
  compose "$tag" logs --tail=50 migrate api worker >&2 || true
  if [[ -n "$previous" ]]; then
    # Images only: this cannot undo a migration, so it fails if $tag already applied one.
    echo "rolling back to $previous" >&2
    start "$previous" || echo "ROLLBACK FAILED: the site may be down" >&2
  fi
  exit 1
fi

echo "$tag" >"$tag_file"
# The image just replaced stays for a week so an old SHA can be redeployed without a pull.
docker image prune --all --force --filter "until=168h" >/dev/null
echo "deployed $tag"
