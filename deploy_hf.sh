#!/usr/bin/env bash
# Redeploy the current `main` to the Hugging Face Space as a clean, history-free
# snapshot. HF rejects PNGs stored as plain git blobs, so we ship an orphan commit
# that drops the (unused) doc images — without touching `main` or GitHub history.
#
# Usage:  ./deploy_hf.sh        (commit your changes on main first)
set -e

git checkout main
git branch -D hf-deploy 2>/dev/null || true

git checkout --orphan hf-deploy
rm -f docs/images/*.png docs/*.png
git add -A
git commit -m "Deploy snapshot"

git push --force space hf-deploy:main

git checkout -f main
git branch -D hf-deploy
echo "✅ Deployed → https://huggingface.co/spaces/Matcry/Rabbook"
