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

