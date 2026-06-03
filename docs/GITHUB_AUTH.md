# GitHub auth (no password in terminal)

GitHub **does not accept account passwords** for `git clone` / `git push` over HTTPS anymore.

Use one of these:

## On your Mac (already working)

You are logged in as **RegularJoe-CEO** via:

```bash
gh auth login
gh auth status
```

Push/pull from Desktop:

```bash
cd ~/Desktop/eRock/_audit_repos/attention-transformer-v2
git push origin main
```

## On RunPod (private repo)

**Do not paste your GitHub password.** Use the Mac helper:

```bash
export RUNPOD_SSH='root@POD_IP -p PORT'
bash scripts/pod_from_mac.sh
```

That uses `gh auth token` from your Mac keyring once per SSH session.

## If your password was shared in chat

1. Change it immediately: https://github.com/settings/security  
2. Enable 2FA: https://github.com/settings/security  
3. Use `gh auth login` or a PAT — never the account password for Git