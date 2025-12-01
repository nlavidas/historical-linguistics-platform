# Smart Autonomous Deployment - Waits for OVH Migration
# This script monitors your OVH instance and deploys automatically when ready

param(
    [string]$VM_IP = "135.125.216.3",
    [string]$VM_USER = "ubuntu",
    [string]$SSH_KEY_PATH = "$env:USERPROFILE\.ssh\id_rsa_ovh",
    [int]$CheckIntervalMinutes = 2
)

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "SMART AUTONOMOUS DEPLOYMENT" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Monitoring OVH instance migration..." -ForegroundColor Yellow
Write-Host "VM: $VM_IP" -ForegroundColor Yellow
Write-Host "Checking every $CheckIntervalMinutes minutes" -ForegroundColor Yellow
Write-Host ""

function Test-VMConnectivity {
    try {
        $null = ssh -i $SSH_KEY_PATH -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$VM_USER@$VM_IP" "echo 'connected'" 2>$null
        return $true
    } catch {
        return $false
    }
}

function Get-VMStatus {
    try {
        ssh -i $SSH_KEY_PATH -o StrictHostKeyChecking=no "$VM_USER@$VM_IP" "echo 'status_check'" 2>$null | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Get-VMResources {
    try {
        $resources = ssh -i $SSH_KEY_PATH -o StrictHostKeyChecking=no "$VM_USER@$VM_IP" "grep MemTotal /proc/meminfo | awk '{print int(\$2/1024/1024) \"GB RAM\"}'; nproc | tr -d '\n'; echo ' vCPU'" 2>$null
        return $resources
    } catch {
        return "Unknown"
    }
}

$connected = $false
$resources = "Checking..."

while (-not $connected) {
    Write-Host "$(Get-Date -Format 'HH:mm:ss') - Checking VM connectivity..." -NoNewline

    if (Test-VMConnectivity) {
        Write-Host " CONNECTED" -ForegroundColor Green
        $connected = $true
        $resources = Get-VMResources
        Write-Host "VM Resources: $resources" -ForegroundColor Green

        # Check if resources are sufficient
        if ($resources -match "(\d+)GB RAM" -and $resources -match "(\d+) vCPU") {
            $ram = [int]$matches[1]
            $cpu = [int]$matches[2]

            if ($ram -ge 32 -and $cpu -ge 8) {
                Write-Host "✓ Resources sufficient for autonomous platform!" -ForegroundColor Green
                break
            } else {
                Write-Host "⚠ Resources may be insufficient ($ram GB RAM, $cpu vCPU)" -ForegroundColor Yellow
                Write-Host "  Recommended: At least 32GB RAM, 8 vCPU" -ForegroundColor Yellow
                $proceed = Read-Host "Proceed anyway? (y/N)"
                if ($proceed -eq "y" -or $proceed -eq "Y") {
                    break
                } else {
                    Write-Host "Aborting deployment. Please upgrade your instance first." -ForegroundColor Red
                    exit 1
                }
            }
        }
    } else {
        Write-Host " NOT READY (migration in progress)" -ForegroundColor Yellow
    }

    Write-Host "Waiting $CheckIntervalMinutes minutes before next check..." -ForegroundColor Gray
    Start-Sleep -Seconds ($CheckIntervalMinutes * 60)
}

Write-Host ""
Write-Host "========================================================================" -ForegroundColor Green
Write-Host "VM IS READY! STARTING AUTONOMOUS DEPLOYMENT" -ForegroundColor Green
Write-Host "========================================================================" -ForegroundColor Green
Write-Host ""

# Now run the complete deployment
& "$PSScriptRoot\COMPLETE_AUTONOMOUS_DEPLOY.ps1"
