# Sugar Knight Share

Bearer-token protected FastAPI file sharing service for `share.sugar-knight.com`.

The service supports upload reservations, upload progress, cancellation, deletion, SHA-256 hashes, public download URLs, and per-file access-token download URLs. Large downloads are authorized by FastAPI and served by nginx through `X-Accel-Redirect`.

## Layout

Production paths on the current host:

```text
/var/www/share-app.sugar-knight.com   application
/srv/share-service/share.db           SQLite metadata database
/srv/share-service/files              uploaded file storage
/etc/systemd/system/share-fastapi-app.service
/etc/nginx/sites-available/share.sugar-knight.com
```

## Configuration

Copy `.env.example` to `.env` and set a strong admin token:

```sh
cp .env.example .env
openssl rand -hex 32
```

Required environment variables:

```text
SHARE_ADMIN_TOKEN=replace-with-secret
SHARE_BASE_URL=https://share.sugar-knight.com
SHARE_DB_PATH=/srv/share-service/share.db
SHARE_STORAGE_DIR=/srv/share-service/files
SHARE_MAX_UPLOAD_BYTES=107374182400
```

Do not commit `.env`. It contains the bearer token used for admin operations.

## Install

Create storage directories:

```sh
mkdir -p /srv/share-service/files /srv/share-service/tmp
chgrp -R www-data /srv/share-service
find /srv/share-service -type d -exec chmod 750 {} +
```

Create the virtualenv and install dependencies:

```sh
python3 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r requirements.txt
```

Install systemd and nginx examples:

```sh
cp deploy/share-fastapi-app.service /etc/systemd/system/share-fastapi-app.service
cp deploy/nginx-share.sugar-knight.com /etc/nginx/sites-available/share.sugar-knight.com
ln -s /etc/nginx/sites-available/share.sugar-knight.com /etc/nginx/sites-enabled/share.sugar-knight.com
systemctl daemon-reload
systemctl enable --now share-fastapi-app.service
nginx -t
systemctl reload nginx
```

The nginx config allows request bodies up to `100G` and disables request buffering for streaming uploads.

## API

Set the admin token in your shell:

```sh
export SHARE_ADMIN_TOKEN='replace-with-secret'
```

Reserve an upload:

```sh
curl -X POST https://share.sugar-knight.com/api/uploads/reserve \
  -H "Authorization: Bearer $SHARE_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"filename":"example.bin","size":123,"content_type":"application/octet-stream","public":true}'
```

The response includes `id`, `upload_token`, `upload_url`, `status_url`, `cancel_url`, and `delete_url`.

Upload content using the returned upload token:

```sh
curl -X PUT "https://share.sugar-knight.com/api/uploads/$UPLOAD_ID/content" \
  -H "Authorization: Bearer $UPLOAD_TOKEN" \
  --data-binary @example.bin
```

Check status, progress, URL, and SHA-256:

```sh
curl "https://share.sugar-knight.com/api/uploads/$UPLOAD_ID" \
  -H "Authorization: Bearer $UPLOAD_TOKEN"
```

Cancel an upload:

```sh
curl -X POST "https://share.sugar-knight.com/api/uploads/$UPLOAD_ID/cancel" \
  -H "Authorization: Bearer $UPLOAD_TOKEN"
```

Delete a file:

```sh
curl -X DELETE "https://share.sugar-knight.com/api/files/$UPLOAD_ID" \
  -H "Authorization: Bearer $SHARE_ADMIN_TOKEN"
```

Create an access-token download URL:

```sh
curl -X POST "https://share.sugar-knight.com/api/files/$UPLOAD_ID/tokens" \
  -H "Authorization: Bearer $SHARE_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"label":"recipient","expires_in_seconds":86400}'
```

Make a file private or public:

```sh
curl -X PATCH "https://share.sugar-knight.com/api/files/$UPLOAD_ID/public" \
  -H "Authorization: Bearer $SHARE_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"public":false}'
```

Authenticated OpenAPI schema and Swagger UI:

```sh
curl -H "Authorization: Bearer $SHARE_ADMIN_TOKEN" \
  https://share.sugar-knight.com/openapi.json
```

`/docs` and `/openapi.json` return `401` without the admin bearer token.

## Operations

Health check:

```sh
curl https://share.sugar-knight.com/health
```

Service status:

```sh
systemctl status share-fastapi-app.service
journalctl -u share-fastapi-app.service -f
```

Validate nginx:

```sh
nginx -t
systemctl reload nginx
```
