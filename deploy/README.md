# Historical Linguistics Platform - OVH Server Deployment

This directory contains deployment files for running the platform 24/7 on an OVH server.

## Quick Start

```bash
# On your OVH server, clone the repository
git clone https://github.com/nlavidas/historical-linguistics-platform.git
cd historical-linguistics-platform

# Run the installation script
sudo ./deploy/install.sh
```

## Services

The platform runs as two systemd services:

### hlp-api.service
- FastAPI REST API server
- Runs on port 8000
- Provides endpoints for corpus management, annotation, valency analysis, etc.

### hlp-scheduler.service
- 24/7 background scheduler
- Collects texts from Perseus, First1KGreek, Gutenberg, PROIEL
- Processes and annotates texts automatically
- Extracts valency patterns
- Runs continuously even when your laptop is closed

## Management Commands

```bash
# View service status
sudo systemctl status hlp-api
sudo systemctl status hlp-scheduler

# View logs
sudo journalctl -u hlp-api -f
sudo journalctl -u hlp-scheduler -f

# Restart services
sudo systemctl restart hlp-api
sudo systemctl restart hlp-scheduler

# Stop services
sudo systemctl stop hlp-api hlp-scheduler

# Start services
sudo systemctl start hlp-api hlp-scheduler
```

## Log Files

- API logs: `/var/log/hlp/api.log`
- API errors: `/var/log/hlp/api-error.log`
- Scheduler logs: `/var/log/hlp/scheduler.log`
- Scheduler errors: `/var/log/hlp/scheduler-error.log`

## Database

The platform uses SQLite for persistent storage:
- Location: `data/hlp_corpus.db`
- All data persists across restarts
- Automatic schema migrations

## API Endpoints

Once running, access the API at:
- API root: http://your-server:8000
- API docs: http://your-server:8000/docs
- Health check: http://your-server:8000/health

## Troubleshooting

If services fail to start:

1. Check logs: `sudo journalctl -u hlp-api -n 50`
2. Verify Python dependencies: `pip3 list | grep -E "fastapi|uvicorn|stanza"`
3. Check permissions: `ls -la /home/ubuntu/historical-linguistics-platform/data`
4. Verify port availability: `sudo netstat -tlnp | grep 8000`

## University of Athens - Nikolaos Lavidas
Funded by Hellenic Foundation for Research and Innovation (HFRI)
