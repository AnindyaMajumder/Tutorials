## Clean existing SSH host key issues

This section helps remove old or incorrect SSH host keys and safely add the correct one.

### 1) Inspect the problematic `known_hosts` line (optional)

```bash
sed -n '9p' ~/.ssh/known_hosts
```

### 2) Remove the old host entry (recommended)

```bash
ssh-keygen -f ~/.ssh/known_hosts -R '<SERVER_IP>'
```

### 3) Fetch the new host key and verify its fingerprint

```bash
# Save the host key to a temporary file
ssh-keyscan -t ed25519 <SERVER_IP> > /tmp/hostkey.pub 2>/dev/null

# Show the SHA256 fingerprint for verification
ssh-keygen -lf -E sha256 /tmp/hostkey.pub
```

* Compare the displayed fingerprint with the one provided by the server provider or administrator.
* Only continue if the fingerprints match.

### 4) Add the verified host key (IP and hostname)

```bash
ssh-keyscan -t ed25519 <SERVER_IP> <SERVER_HOSTNAME> >> ~/.ssh/known_hosts
chmod 600 ~/.ssh/known_hosts
```

### 5) Reconnect via SSH

```bash
ssh <SSH_USER>@<SERVER_IP>
```

#### Quick alternative (temporary, less secure)

> Use only for short-term testing.

```bash
ssh -o StrictHostKeyChecking=no <SSH_USER>@<SERVER_IP>
```

---

## VPS basic setup

### Connect to the server

```bash
ssh <SSH_USER>@<SERVER_IP>
```

### Update system and install common tools

```bash
apt update && apt upgrade -y
apt install -y git curl wget nano ufw
```

### Install Docker

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
apt install -y docker-compose-plugin

docker --version
docker compose version
```

### Configure firewall (UFW)

```bash
# Allow SSH
ufw allow 22

# Allow HTTP and HTTPS
ufw allow 80
ufw allow 443

# Allow application port (adjust if needed)
ufw allow <APP_PORT>

# Enable firewall
ufw enable
```

---

## Clone a Git repository

### Public repository

```bash
git clone <REPOSITORY_URL> .
```

### Private repository (using deploy key)

#### Generate a deploy key on the server

```bash
ssh-keygen -t ed25519 -C "server-deploy-key" -f ~/.ssh/repo_deploy -N ""

cat ~/.ssh/repo_deploy.pub
```

#### Add the deploy key to the Git hosting platform

* Go to the repository settings
* Add a **Deploy Key**
* Paste the public key
* Enable write access only if required

#### Configure SSH to use the deploy key

```bash
nano ~/.ssh/config
```

```text
Host git-host
    HostName git-host
    User git
    IdentityFile ~/.ssh/repo_deploy
    IdentitiesOnly yes
```

```bash
chmod 600 ~/.ssh/config
```

#### Test connection

```bash
ssh -T git@git-host
```

#### Clone via SSH

```bash
git clone git@git-host:<ORG>/<REPO>.git
```

---

## Set up server access for automated deployments

### Generate an SSH key for CI/CD

```bash
ssh-keygen -t ed25519 -C "ci-deploy" -f ~/.ssh/ci_deploy -N ""

cat ~/.ssh/ci_deploy.pub
```

### Add public key to the server

```bash
nano ~/.ssh/authorized_keys
```

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

### Configure CI/CD secrets

Add the following secrets to your pipeline:

```
SSH_HOST=<SERVER_IP>
SSH_USERNAME=<SSH_USER>
SSH_PRIVATE_KEY=<PRIVATE_KEY_CONTENT>
SSH_PORT=22
DEPLOY_PATH=<DEPLOY_DIRECTORY>
```

Update the deployment workflow file accordingly.

---

## Nginx reverse proxy and HTTPS setup

> Assumes the application runs on a local port.

### Test Nginx configuration

```bash
sudo nginx -t
```

### Verify the local application

```bash
curl http://localhost:<APP_PORT>
```

### Edit Nginx site configuration

```bash
sudo nano /etc/nginx/sites-available/default
```

```nginx
server {
    listen 80;
    server_name <DOMAIN_NAME>;

    location / {
        proxy_pass http://localhost:<APP_PORT>;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_cache_bypass $http_upgrade;
    }
}
```

### Validate and reload Nginx

```bash
sudo nginx -t
sudo systemctl reload nginx
```

---

## Enable HTTPS with Certbot

### Install Certbot

```bash
sudo apt install certbot python3-certbot-nginx
```

### Obtain and install SSL certificate

```bash
sudo certbot --nginx -d <DOMAIN_NAME>
```

### Test automatic renewal

```bash
sudo certbot renew --dry-run
```

> Certificates are automatically renewed. Manual renewal is rarely needed.
