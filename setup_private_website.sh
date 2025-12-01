#!/bin/bash
# Password-Protected Private Website Setup
# Creates a static site from research exports and protects with HTTP auth

SITE_DIR="/var/www/corpus-platform"
DOMAIN="corpus-platform.nlavid.as"  # Replace with your domain

# Install required packages
sudo apt update
sudo apt install -y nginx apache2-utils mkdocs python3-mkdocs

# Create MkDocs site from research exports
cd ~/corpus_platform

# Generate MkDocs config
cat > mkdocs.yml << EOF
site_name: Diachronic Corpus Platform
site_description: Autonomous Historical Linguistics Research
site_author: nlavidas
copyright: '© 2025 Autonomous Platform'

nav:
  - Home: index.md
  - Corpus Overview: corpus.md
  - Research Reports: reports.md
  - Agent Reports: agent-reports.md
  - Visual Reports: visual-reports.md

theme:
  name: material
  palette:
    primary: blue
    accent: light-blue
  features:
    - navigation.tabs
    - navigation.sections
    - toc.integrate

plugins:
  - search
  - mkdocstrings
EOF

# Create documentation pages
cat > docs/index.md << EOF
# Diachronic Corpus Platform

Welcome to the autonomous historical linguistics platform.

## Features

- **Autonomous Collection**: 24/7 text collection from APIs
- **Linguistic Analysis**: Stanza-based lemmatization and parsing
- **Multi-Agent System**: CrewAI-powered research agents
- **Vector Search**: ChromaDB semantic search
- **Web Interface**: Flask-based control panel

## Access

- **Web Control Panel**: [http://your-vm-ip](http://your-vm-ip)
- **VS Code IDE**: [http://your-vm-ip/ide](http://your-vm-ip/ide) (password: historical_linguistics_2025)
EOF

# Build static site
mkdocs build --clean

# Deploy to nginx
sudo mkdir -p $SITE_DIR
sudo cp -r site/* $SITE_DIR/
sudo chown -R www-data:www-data $SITE_DIR

# Create password file
sudo htpasswd -cb /etc/nginx/.htpasswd admin "historical_linguistics_2025"

# Configure nginx with auth
sudo tee /etc/nginx/sites-available/corpus-platform-site > /dev/null << EOF
server {
    listen 80;
    server_name $DOMAIN;

    # SSL redirect (if you have certbot)
    # return 301 https://\$server_name\$request_uri;

    # Root directory
    root $SITE_DIR;
    index index.html;

    # Password protection
    auth_basic "Corpus Platform - Private Access";
    auth_basic_user_file /etc/nginx/.htpasswd;

    location / {
        try_files \$uri \$uri/ =404;
    }

    # API proxy to Flask app
    location /api/ {
        auth_basic off;  # No auth for API calls
        proxy_pass http://localhost:8000/api/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOF

# Enable site
sudo ln -sf /etc/nginx/sites-available/corpus-platform-site /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# Set up automated rebuild
echo "0 3 * * * cd ~/corpus_platform && mkdocs build --clean && sudo cp -r site/* $SITE_DIR/" | crontab -

echo "========================================================================"
echo "Private Password-Protected Website Setup Complete!"
echo "========================================================================"
echo ""
echo "Site URL: http://$DOMAIN"
echo "Username: admin"
echo "Password: historical_linguistics_2025"
echo ""
echo "Features:"
echo "  - MkDocs-generated static site"
echo "  - HTTP Basic Authentication"
echo "  - Automated nightly rebuilds"
echo "  - API proxy to Flask backend"
echo ""
echo "To rebuild manually:"
echo "  mkdocs build --clean && sudo cp -r site/* $SITE_DIR/"
echo "========================================================================"
