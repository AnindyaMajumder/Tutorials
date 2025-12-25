##  Clean your any exisitng ssh blocking to your machine

1) Inspect the offending known_hosts line (optional)
```bash
sed -n '9p' ~/.ssh/known_hosts
```

2) Remove the old entry (recommended; this is the SSH-provided command)
```bash
ssh-keygen -f ~/.ssh/known_hosts -R '195.35.56.142'
```

3) Fetch the new host key and compute its SHA256 fingerprint (do this before adding it)
```bash
# Save the host key to a temp file
ssh-keyscan -t ed25519 195.35.56.142 > /tmp/hostkey.pub 2>/dev/null

# Show the SHA256 fingerprint for verification
ssh-keygen -lf -E sha256 /tmp/hostkey.pub
```

- The fingerprint you posted from the SSH warning was:
  SHA256:2yovb6F0QtHZ6mRqcUeV41BZZLLl+z3NutDRRJmJWiU
- Compare the output of the `ssh-keygen -lf` command with that string.
- If you have access to Hostinger control panel (or Hostinger support), also verify the fingerprint there before trusting it.

4) If the fingerprint matches what you expect (or support confirms), add the host key for both the IP and hostname:
```bash
# Add both the IP and hostname entry to your known_hosts
ssh-keyscan -t ed25519 195.35.56.142 srv1218075.hstgr.cloud >> ~/.ssh/known_hosts
# Secure the file permissions
chmod 600 ~/.ssh/known_hosts
```

5) Reconnect via SSH
```bash
ssh root@195.35.56.142
```

Quick alternative (temporary, less secure)
- If you need an immediate, temporary bypass (not recommended for production):
```bash
ssh -o StrictHostKeyChecking=no root@195.35.56.142
```
This will accept the host key for that session without saving it. Do not use this habitually.

## VPS Setup
Connect to the VPS through SSH with IPv4 and root password
```bash
ssh root@195.35.56.142
```
Update System & Install Dependencies
```bash
apt update && apt upgrade -y
apt install -y git curl wget nano ufw

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
apt install -y docker-compose-plugin
docker --version
docker compose version
```
Configure Firewall
```bash
# Allow SSH
ufw allow 22

# Allow HTTP/HTTPS
ufw allow 80
ufw allow 443

# Allow your app port (check your docker-compose.yml for the exposed port)
ufw allow 8000

# Enable firewall
ufw enable
```
Clone Your Repository (if public)
```bash
git clone https://github.com/AnindyaMajumder/Learn-English-AI.git .
```
### If private repository
Generate a Deploy Key on VPS
```bash
ssh-keygen -t ed25519 -C "hostinger-vps-deploy" -f ~/.ssh/github_deploy -N ""

# View the public key
cat ~/.ssh/github_deploy.pub
```
Add Deploy Key to GitHub
Go to: https://github.com/AnindyaMajumder/Learn-English-AI/settings/keys
Click "Add deploy key" -> Paste the public key -> ✅ Check "Allow write access" (optional, not needed for pull-only) -> Click "Add key"

Configure SSH on VPS to Use the Key
```bash
nano ~/.ssh/config
```
Add this content:
```
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/github_deploy
    IdentitiesOnly yes
```
Set permissions
```bash
chmod 600 ~/.ssh/config
```
Test Github Connections
```bash
ssh -T git@github.com
```
> Are you sure you want to continue connecting (yes/no/[fingerprint])? yes

Clone Repository through SSH
```bash
git@github.com:AnindyaMajumder/Learn-English-AI.git
```

## Up your repository
Enter the directory of clonned repository
Set Up SSH Key for GitHub Actions
```bash
ssh-keygen -t ed25519 -C "github-actions-deploy" -f ~/.ssh/hostinger_deploy -N ""

cat ~/.ssh/hostinger_deploy
cat ~/.ssh/hostinger_deploy.pub
```
Add Public Key to VPS
```bash
# Add the public key to authorized_keys
nano ~/.ssh/authorized_keys
# Paste the public key content and save

# Set correct permissions
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

### Configure GitHub Secrets
Add these secrets:
```
SSH_HOST	195.35.56.142
SSH_USERNAME	root
SSH_PRIVATE_KEY	(paste entire private key including BEGIN/END lines)
SSH_PORT	22
DEPLOY_PATH	/home/ColWords
```
Lastly update the `.github/workflow/deploy.yml`

## Set up nginx for port forwarding 
> (Assuming the server is running on different port)

Check nginx configuration for reverse proxy
```bash
sudo nginx -t
```
Check if the local server is running
```bash
curl http://localhost:8000
```
Edit the Nginx Configuration
```bash
sudo nano /etc/nginx/sites-available/default
```
Copy and paste this script
```
server {
    listen 80;
    server_name chat.eflip.com;  
    location / {
        proxy_pass http://localhost:8000;
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
Check the syntax and configuration again
```bash
sudo nginx -t
```
Install Certbot
```bash
sudo apt install certbot python3-certbot-nginx
```
Run certbot with the domain/subdomain to active SSL
```bash
sudo certbot --nginx -d chat.eflip.com
```
Baaannnggg!! HTTPS enabled!!
> Let's Encrypt certificates are valid for 90 days. When you installed Certbot (using apt install), it automatically set up a background timer to check your certificates twice a day and renew any that are close to expiring. You usually don't need to do anything manually.

Check if Auto-Renewal is working
```bash
sudo certbot renew --dry-run
```
If auto-renewal failed, try manually
```bash
sudo certbot renew
```
