#!/usr/bin/env bash
set -euo pipefail

cd "$APP_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y python3 python3-venv python3-pip
fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

cat > .env <<ENVFILE
OPENAI_API_KEY=$OPENAI_API_KEY
OPENAI_MODEL=$OPENAI_MODEL
SEARCH_PROVIDER=$SEARCH_PROVIDER
TAVILY_API_KEY=$TAVILY_API_KEY
SERPAPI_API_KEY=$SERPAPI_API_KEY
PORT=8000
ENVFILE

sed \
  -e "s|__APP_DIR__|$APP_DIR|g" \
  -e "s|__APP_USER__|$USER|g" \
  deploy/fakechecker.service.template | sudo tee /etc/systemd/system/fakechecker.service >/dev/null

sudo systemctl daemon-reload
sudo systemctl enable fakechecker.service
sudo systemctl restart fakechecker.service

if [ -n "${APP_DOMAIN:-}" ]; then
  if ! command -v nginx >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y nginx
  fi

  sed \
    -e "s|__APP_DOMAIN__|$APP_DOMAIN|g" \
    -e "s|__APP_DIR__|$APP_DIR|g" \
    deploy/nginx.fakechecker.conf.template | sudo tee /etc/nginx/sites-available/fakechecker >/dev/null

  sudo ln -sf /etc/nginx/sites-available/fakechecker /etc/nginx/sites-enabled/fakechecker
  sudo nginx -t
  sudo systemctl reload nginx
fi
