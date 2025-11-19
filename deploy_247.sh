#!/bin/bash

# 24/7 Deployment Script for Corpus Platform
# This script ensures the platform runs continuously with monitoring and auto-recovery

# Configuration
PROJECT_DIR="/home/ubuntu/corpus_platform"  # Update this path for your server
LOG_DIR="$PROJECT_DIR/logs"
BACKUP_DIR="$PROJECT_DIR/backups"
DOCKER_COMPOSE="docker-compose -f $PROJECT_DIR/docker-compose.yml"

# Create necessary directories
mkdir -p "$LOG_DIR" "$BACKUP_DIR"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/deployment.log"
}

# Function to check if a service is running
is_running() {
    $DOCKER_COMPOSE ps -q "$1" | grep -q .
}

# Function to start services
start_services() {
    log "Starting Corpus Platform services..."
    cd "$PROJECT_DIR" || exit 1
    
    # Pull latest changes if using version control
    # git pull origin main  # Uncomment if using git
    
    # Start services
    $DOCKER_COMPOSE up -d --build
    
    # Verify services started
    if is_running "corpus-platform" && is_running "corpus-pipeline"; then
        log "Services started successfully"
    else
        log "ERROR: Failed to start services"
        return 1
    fi
}

# Function to stop services
stop_services() {
    log "Stopping Corpus Platform services..."
    cd "$PROJECT_DIR" || exit 1
    $DOCKER_COMPOSE down
}

# Function to restart services
restart_services() {
    log "Restarting services..."
    stop_services
    sleep 5
    start_services
}

# Function to create backup
create_backup() {
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local backup_file="$BACKUP_DIR/corpus_backup_$timestamp.tar.gz"
    
    log "Creating backup at $backup_file"
    
    # Stop services before backup
    stop_services
    
    # Create backup
    tar -czf "$backup_file" -C "$PROJECT_DIR" data corpus_platform.db
    
    # Restart services
    start_services
    
    log "Backup completed: $backup_file"
}

# Function to monitor services
monitor_services() {
    while true; do
        if ! is_running "corpus-platform"; then
            log "WARNING: corpus-platform is not running, restarting..."
            start_services
        fi
        
        if ! is_running "corpus-pipeline"; then
            log "WARNING: corpus-pipeline is not running, restarting..."
            start_services
        fi
        
        # Check disk space
        local disk_usage=$(df -h / | awk 'NR==2 {print $5}' | tr -d '%')
        if [ "$disk_usage" -gt 90 ]; then
            log "WARNING: Disk usage is at ${disk_usage}%, cleaning up..."
            docker system prune -f
        fi
        
        # Check memory usage
        local mem_usage=$(free -m | awk 'NR==2{printf "%.1f", $3*100/$2}')
        if (( $(echo "$mem_usage > 90" | bc -l) )); then
            log "WARNING: High memory usage (${mem_usage}%), restarting services..."
            restart_services
        fi
        
        # Create daily backup at 2 AM
        if [ "$(date +%H%M)" = "0200" ]; then
            create_backup
        fi
        
        # Sleep for 5 minutes before next check
        sleep 300
    done
}

# Main execution
case "$1" in
    start)
        start_services
        monitor_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    backup)
        create_backup
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|backup}"
        exit 1
        ;;
esac
