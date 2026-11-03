# Deploying Hyperplane

A push to `main` runs CI. If CI passes, the `Deploy` workflow builds images, pushes them to GHCR
and rolls them out to the server over SSH. Other branches never deploy.

```
browser → Cloudflare → Tunnel → cloudflared → web (nginx) → api → postgres, redis, garage
                                                                  worker ↗
```

The server publishes no ports for the app. Only SSH needs to be reachable.

## Requirements

A Linux VPS with root access and Docker: at least 2 vCPU, 4 GB RAM (8 GB recommended, since the
worker alone may use 2 GB) and 50 GB of disk. Shared web hosting cannot run this stack.

## 1. Prepare the server

As root, on Ubuntu or Debian:

```bash
curl -fsSL https://get.docker.com | sh
cat > /etc/docker/daemon.json <<'EOF'
{ "log-driver": "json-file", "log-opts": { "max-size": "50m", "max-file": "3" } }
EOF
systemctl restart docker

adduser --disabled-password --gecos "" deploy
usermod -aG docker deploy              # docker access is root-equivalent; keep this key CI-only
install -d -o deploy -g deploy /opt/hyperplane

ufw allow OpenSSH && ufw --force enable
```

## 2. Create the deploy key

On your own machine:

```bash
ssh-keygen -t ed25519 -f hyperplane-deploy -N "" -C "github-deploy"
ssh-copy-id -i hyperplane-deploy.pub deploy@<server-ip>
ssh-keyscan -t ed25519 <server-ip>     # the value of DEPLOY_KNOWN_HOSTS
```

If `ssh-copy-id` cannot log in as `deploy`, append the `.pub` file to
`/home/deploy/.ssh/authorized_keys` as root (mode 600, owned by `deploy`).

## 3. Create the Cloudflare Tunnel

In Cloudflare, go to **Zero Trust → Networks → Tunnels → Create a tunnel → Cloudflared**, name it
`hyperplane`, and copy the token from the install command. Then add a **public hostname**:

| Hostname | Service |
| --- | --- |
| the site's hostname | `HTTP`, `web:80` |

If the hostname already has an A or CNAME record, Cloudflare asks you to remove it first. On
`traintestval.com`, that record is what serves the current WordPress site, so that step is the
cutover. Leave MX and TXT records alone. Do this last if you want no gap (step 6).

In **Speed → Settings**, turn Rocket Loader off, because it rewrites script loading.

## 4. Write the server's secrets

```bash
# as deploy, on the server, after copying infra/production.env.example to /opt/hyperplane/.env
chmod 600 /opt/hyperplane/.env
nano /opt/hyperplane/.env
```

## 5. Configure GitHub

In the repository, go to **Settings → Environments → New environment**, name it `production`, and
set **Deployment branches and tags** to the selected branch `main`. Add these environment secrets:

| Secret | Value |
| --- | --- |
| `DEPLOY_HOST` | server IP |
| `DEPLOY_USER` | `deploy` |
| `DEPLOY_SSH_KEY` | contents of the private `hyperplane-deploy` file |
| `DEPLOY_KNOWN_HOSTS` | output of `ssh-keyscan` |

Then delete the local private key file. Protecting `main` under **Settings → Branches** is also
worthwhile, so that every change reaches it through a passing pull request.

## 6. First deploy

Merge `dev` into `main` and push. Watch the **Actions** tab: CI runs first, then `Deploy`. When it
is green, check the tunnel shows **Healthy** and do the DNS cutover from step 3 if still pending.

```bash
curl -s https://<hostname>/api/v1/health    # db, redis and storage all reachable
```

## Operations

```bash
# on the server, as deploy, from /opt/hyperplane/infra
alias hp='IMAGE_TAG=$(cat ../.deployed-tag) docker compose --env-file ../.env -f docker-compose.yml -f docker-compose.prod.yml'
hp ps
hp logs --tail=100 api worker
./deploy.sh <older-commit-sha>             # manual rollback (needs a `docker login ghcr.io` if the image was pruned)
```

**Failed deploys.** `deploy.sh` waits for every health check and restores the previous images if
one fails. It cannot undo a database migration, so a release that applied one and then failed has
to be fixed forward with a new commit.

**Backups.** There are none yet. Postgres and Garage hold only anonymous datasets and runs, which
expire after 7 days. Learning content ships inside the image.
