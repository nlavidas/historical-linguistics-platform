#!/bin/bash
# EMERGENCY DISK CLEANUP - Run this FIRST on your VM

echo "=== EMERGENCY DISK CLEANUP ==="
echo "Current disk usage:"
df -h /

echo ""
echo "=== FINDING LARGE FILES ==="
echo "Files over 100MB:"
find /home/ubuntu -type f -size +100M -exec ls -lh {} \; 2>/dev/null | head -10

echo ""
echo "=== DIRECTORY SIZES ==="
du -sh /home/ubuntu/* 2>/dev/null | sort -hr | head -10

echo ""
echo "=== SAFE CLEANUP OPTIONS ==="
echo "1. Remove old log files (>30 days):"
find /home/ubuntu -name "*.log" -type f -mtime +30 -exec rm -f {} \; 2>/dev/null && echo "Old logs cleaned"

echo "2. Remove old cycle reports (>7 days):"
find /home/ubuntu -name "cycle_*.txt" -type f -mtime +7 -exec rm -f {} \; 2>/dev/null && echo "Old cycle reports cleaned"

echo "3. Remove old HTML reports (>7 days):"
find /home/ubuntu -name "*.html" -type f -mtime +7 -exec rm -f {} \; 2>/dev/null && echo "Old HTML reports cleaned"

echo "4. Clean package manager cache:"
sudo apt autoremove -y && sudo apt autoclean -y && echo "Package cache cleaned"

echo ""
echo "=== AFTER CLEANUP ==="
df -h /

echo ""
echo "If still full, you can:"
echo "- Delete unused database backups"
echo "- Move large files to external storage"
echo "- Contact OVH to increase disk size"
echo ""
echo "Once disk has >5GB free, run the deployment again!"
