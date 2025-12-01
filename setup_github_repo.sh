#!/bin/bash
# GitHub Integration Script
# Creates private repo and sets up automated syncing

REPO_NAME="diachronic-corpus-platform"
GITHUB_USER="nlavidas"

# Check if git is initialized
if [ ! -d .git ]; then
    git init
    echo "Git repository initialized"
fi

# Add remote (assuming SSH key is set up)
git remote add origin "git@github.com:$GITHUB_USER/$REPO_NAME.git" 2>/dev/null || git remote set-url origin "git@github.com:$GITHUB_USER/$REPO_NAME.git"

# Create .gitignore for sensitive data
cat > .gitignore << EOF
# Database and sensitive data
corpus_platform.db
*.db
*.sqlite
*.sqlite3

# Logs and temporary files
*.log
__pycache__/
*.pyc
.cache/
temp/
old_*/

# Model files (large)
*.gguf
*.bin
*.pth
models/

# SSH keys and credentials
*.key
*.pem
.env
config/secrets.json

# VS Code
.vscode/settings.json
EOF

# Initial commit
git add .
git reset -- research_exports/  # Don't commit large data files initially
git commit -m "Initial autonomous platform setup

- Flask web control panel
- Ollama + Qwen2.5-Coder integration
- CrewAI multi-agent system
- ChromaDB vector search
- Stanza linguistic processing
- Autonomous VM deployment
- Research agent with ML suggestions"

# Push to GitHub
echo "Pushing to GitHub..."
git branch -M main
git push -u origin main

echo "========================================================================"
echo "GitHub Repository Setup Complete!"
echo "========================================================================"
echo ""
echo "Repository: https://github.com/$GITHUB_USER/$REPO_NAME"
echo ""
echo "To sync future changes:"
echo "  git add . && git commit -m 'Update' && git push"
echo ""
echo "For automated nightly sync:"
echo "  crontab -e"
echo "  Add: 0 2 * * * cd ~/corpus_platform && git add . && git commit -m \"Nightly sync \$(date)\" && git push"
echo "========================================================================"
