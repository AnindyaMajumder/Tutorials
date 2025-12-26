## Clean existing SSH host key issues

> **Placeholders & examples used in this guide**
>
> * `<SERVER_IP>` → `203.0.113.10`
> * `<SERVER_HOSTNAME>` → `server.example.com`
> * `<SSH_USER>` → `root`
> * `<APP_PORT>` → `8000`
> * `<REPOSITORY_URL>` → `https://github.com/example-org/example-repo.git`
> * `<DOMAIN_NAME>` → `app.example.com`
> * `<ORG>` → `example-org`
> * `<REPO>` → `example-repo`
> * `<DEPLOY_DIRECTORY>` → `/var/www/app`
>
> Replace the examples with real values when applying to a real server.

## Clean existing SSH host key issues

This section helps remove old or incorrect SSH host keys and safely add the correct one.

### 1) Inspect the problematic `known_hosts` line (optional)

This step checks whether a specific saved SSH key is causing the warning. It is only for inspection and does not change anything.

```bash
sed -n '9p' ~/.ssh/known_hosts
```

### 2) Remove the old host entry (recommended)

Here the outdated or mismatched SSH key is removed so a fresh, correct key can be added safely.

```bash
ssh-keygen -f ~/.ssh/known_hosts -R '<SERVER_IP>'
# example:
# ssh-keygen -f ~/.ssh/known_hosts -R '203.0.113.10'
```

### 3) Fetch the new host key and verify its fingerprint

This step retrieves the server’s public SSH key and lets you confirm its identity before trusting it.

```bash
# Save the host key to a temporary file
ssh-keyscan -t ed25519 <SERVER_IP> > /tmp/hostkey.pub 2>/dev/null
# example:
# ssh-keyscan -t ed25519 203.0.113.10 > /tmp/hostkey.pub 2>/dev/null

# Show the SHA256 fingerprint for verification
ssh-keygen -lf -E sha256 /tmp/hostkey.pub
```

* Compare the displayed fingerprint with the one provided by the server provider or administrator.
* Only continue if the fingerprints match.

### 4) Add the verified host key (IP and hostname)

After verification, the trusted key is permanently stored so future SSH connections do not show warnings.

```bash
ssh-keyscan -t ed25519 <SERVER_IP> <SERVER_HOSTNAME> >> ~/.ssh/known_hosts
# example:
# ssh-keyscan -t ed25519 203.0.113.10 server.example.com >> ~/.ssh/known_hosts
chmod 600 ~/.ssh/known_hosts
```

### 5) Reconnect via SSH

Now that the correct key is saved, you can connect normally without security warnings.

```bash
ssh <SSH_USER>@<SERVER_IP>
# example:
# ssh root@203.0.113.10
```

#### Quick alternative (temporary, less secure)

> Use only for short-term testing.

```bash
ssh -o StrictHostKeyChecking=no <SSH_USER>@<SERVER_IP>
```

---

## VPS basic setup

### Connect to the server

This opens a secure remote shell session on the VPS so it can be configured.

```bash
ssh <SSH_USER>@<SERVER_IP
# example:
# ssh root@203.0.113.10
```

### Update system and install common tools

Keeping the system updated ensures security patches are applied and installs basic utilities needed later.

```bash
apt update && apt upgrade -y
apt install -y git curl wget nano ufw
```

### Install Docker

Docker is installed to run applications in containers, making deployments consistent and repeatable.

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
apt install -y docker-compose-plugin

docker --version
docker compose version
```

### Configure firewall (UFW)

The firewall restricts network access so only the required ports are reachable from the internet.

```bash
# Allow SSH
ufw allow 22

# Allow HTTP and HTTPS
ufw allow 80
ufw allow 443

# Allow application port (adjust if needed)
ufw allow <APP_PORT>
# example:
# ufw allow 8000

# Enable firewall
ufw enable
```

---

## Clone a Git repository

### Public repository

This pulls the project source code when it is publicly accessible.

```bash
git clone <REPOSITORY_URL> .
# example:
# git clone https://github.com/example-org/example-repo.git .
```

### Private repository (using deploy key)

A deploy key allows secure, password-less access to private repositories from the server.

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
# example:
# git clone git@git-host:example-org/example-repo.git
```

---

## Set up server access for automated deployments

### Generate an SSH key for CI/CD

This key is used by automation tools to securely connect to the server during deployments.

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

This checks the Nginx configuration file for syntax errors before applying changes.

```bash
sudo nginx -t
```

### Verify the local application

This confirms the application is running locally before exposing it through Nginx.

```bash
curl http://localhost:<APP_PORT>
# example:
# curl http://localhost:8000
```

### Edit Nginx site configuration

```bash
sudo nano /etc/nginx/sites-available/default
```

```nginx
server {
    listen 80;
    server_name <DOMAIN_NAME>;
    # example:
    # server_name app.example.com;

    location / {
        proxy_pass http://localhost:<APP_PORT>;
        # example:
        # proxy_pass http://localhost:8000;
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

Certbot is used to obtain and manage free SSL certificates.

```bash
sudo apt install certbot python3-certbot-nginx
```

### Obtain and install SSL certificate

This step enables HTTPS by automatically configuring Nginx with a valid SSL certificate.

```bash
sudo certbot --nginx -d <DOMAIN_NAME>
# example:
# sudo certbot --nginx -d app.example.com
```

### Test automatic renewal

```bash
sudo certbot renew --dry-run
```

> Certificates are automatically renewed. Manual renewal is rarely needed.
> Test your domain accessibility from https://globalping.io/
