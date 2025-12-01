#!/bin/bash
# Install code-server (open-source VS Code for web)
curl -fsSL https://code-server.dev/install.sh | sh

# Create config directory
mkdir -p ~/.config/code-server

# Configure code-server
cat > ~/.config/code-server/config.yaml << EOF
bind-addr: 0.0.0.0:8080
auth: password
password: historical_linguistics_2025
cert: false
EOF

# Start code-server
nohup code-server --bind-addr 0.0.0.0:8080 > /tmp/code-server.log 2>&1 &

echo "VS Code Server setup complete"
echo "Access at: http://localhost:8080"
echo "Password: historical_linguistics_2025"
