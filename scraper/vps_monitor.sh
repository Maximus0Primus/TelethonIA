#!/bin/bash
# VPS Monitor — lightweight cron script (runs every 5 min)
# Checks RAM, disk, swap, scraper health → alerts Telegram + pushes metrics to Supabase
# Designed to use <1% CPU and <5MB RAM

set -euo pipefail

# Load env
ENV_FILE="/opt/TelethonIA/scraper/.env"
if [ -f "$ENV_FILE" ]; then
    export $(grep -E '^(MONITOR_BOT_TOKEN|MONITOR_CHAT_ID|SUPABASE_URL|SUPABASE_SERVICE_ROLE_KEY)=' "$ENV_FILE" | xargs)
fi

# ── Thresholds ──
RAM_WARN_PCT=80
RAM_CRIT_PCT=90
DISK_WARN_PCT=80
DISK_CRIT_PCT=90
SWAP_WARN_MB=500
SCRAPER_RSS_WARN_MB=1500
LOAD_WARN=3.0

# State file for dedup (don't spam same alert)
STATE_DIR="/tmp/vps_monitor"
mkdir -p "$STATE_DIR"

# ── Helpers ──
send_telegram() {
    local text="$1"
    [ -z "${MONITOR_BOT_TOKEN:-}" ] && return
    [ -z "${MONITOR_CHAT_ID:-}" ] && return
    curl -s -X POST "https://api.telegram.org/bot${MONITOR_BOT_TOKEN}/sendMessage" \
        -H "Content-Type: application/json" \
        -d "{\"chat_id\":\"${MONITOR_CHAT_ID}\",\"text\":\"${text}\",\"parse_mode\":\"HTML\",\"disable_web_page_preview\":true}" \
        > /dev/null 2>&1 || true
}

alert_once() {
    # Only alert if state file doesn't exist or is >1h old (re-alert every hour)
    local key="$1"
    local msg="$2"
    local state_file="${STATE_DIR}/${key}"
    if [ -f "$state_file" ]; then
        local age=$(( $(date +%s) - $(stat -c %Y "$state_file" 2>/dev/null || echo 0) ))
        [ "$age" -lt 3600 ] && return  # already alerted <1h ago
    fi
    send_telegram "$msg"
    touch "$state_file"
}

clear_alert() {
    rm -f "${STATE_DIR}/$1" 2>/dev/null || true
}

# ── Collect metrics ──

# RAM
read RAM_TOTAL RAM_USED RAM_AVAIL <<< $(free -m | awk '/^Mem:/{print $2, $3, $7}')
RAM_PCT=$(( RAM_USED * 100 / RAM_TOTAL ))

# Swap
SWAP_USED=$(free -m | awk '/^Swap:/{print $3}')

# Disk
read DISK_USED_GB DISK_TOTAL_GB DISK_PCT <<< $(df -BG / | awk 'NR==2{gsub(/G/,"",$3); gsub(/G/,"",$2); gsub(/%/,"",$5); print $3, $2, $5}')

# Load
LOAD_1M=$(cat /proc/loadavg | awk '{print $1}')
LOAD_5M=$(cat /proc/loadavg | awk '{print $2}')

# CPU (from /proc/stat — instant, no process overhead)
read -r _ u1 n1 s1 i1 _ < /proc/stat
sleep 1
read -r _ u2 n2 s2 i2 _ < /proc/stat
TOTAL=$(( (u2+n2+s2+i2) - (u1+n1+s1+i1) ))
IDLE=$(( i2 - i1 ))
CPU_PCT=$(( TOTAL > 0 ? (TOTAL - IDLE) * 100 / TOTAL : 0 ))

# Journal size
JOURNAL_MB=$(du -sm /var/log/journal/ 2>/dev/null | awk '{print $1}' || echo 0)

# Scraper process
SCRAPER_PID=$(systemctl show kol-scraper -p MainPID --value 2>/dev/null || echo 0)
SCRAPER_RSS_MB=0
SCRAPER_CPU=0
SCRAPER_UPTIME_S=0
SCRAPER_ACTIVE="inactive"

if [ "$SCRAPER_PID" -gt 0 ] 2>/dev/null; then
    SCRAPER_RSS_MB=$(ps -o rss= -p "$SCRAPER_PID" 2>/dev/null | awk '{printf "%d", $1/1024}' || echo 0)
    SCRAPER_CPU=$(ps -o %cpu= -p "$SCRAPER_PID" 2>/dev/null | awk '{print $1}' || echo 0)
    # Uptime in seconds
    SCRAPER_START=$(date -d "$(systemctl show kol-scraper -p ActiveEnterTimestamp --value)" +%s 2>/dev/null || echo 0)
    NOW=$(date +%s)
    SCRAPER_UPTIME_S=$(( NOW - SCRAPER_START ))
    SCRAPER_ACTIVE="active"
fi

# ── Check & Alert ──

ALERTS=""

# RAM
if [ "$RAM_PCT" -ge "$RAM_CRIT_PCT" ]; then
    alert_once "ram_crit" "<b>RAM CRITICAL: ${RAM_PCT}%</b>
Used: ${RAM_USED}M / ${RAM_TOTAL}M
Available: ${RAM_AVAIL}M"
elif [ "$RAM_PCT" -ge "$RAM_WARN_PCT" ]; then
    alert_once "ram_warn" "<b>RAM WARNING: ${RAM_PCT}%</b>
Used: ${RAM_USED}M / ${RAM_TOTAL}M"
else
    clear_alert "ram_warn"
    clear_alert "ram_crit"
fi

# Swap
if [ "${SWAP_USED:-0}" -ge "$SWAP_WARN_MB" ]; then
    alert_once "swap_warn" "<b>SWAP WARNING: ${SWAP_USED}M used</b>
RAM may be under pressure."
else
    clear_alert "swap_warn"
fi

# Disk
if [ "$DISK_PCT" -ge "$DISK_CRIT_PCT" ]; then
    alert_once "disk_crit" "<b>DISK CRITICAL: ${DISK_PCT}%</b>
Used: ${DISK_USED_GB}G / ${DISK_TOTAL_GB}G"
elif [ "$DISK_PCT" -ge "$DISK_WARN_PCT" ]; then
    alert_once "disk_warn" "<b>DISK WARNING: ${DISK_PCT}%</b>
Used: ${DISK_USED_GB}G / ${DISK_TOTAL_GB}G"
else
    clear_alert "disk_warn"
    clear_alert "disk_crit"
fi

# Scraper down
if [ "$SCRAPER_ACTIVE" != "active" ] || [ "$SCRAPER_PID" -le 0 ] 2>/dev/null; then
    alert_once "scraper_down" "<b>SCRAPER DOWN</b>
kol-scraper.service is not running!
Last check: $(date -u '+%Y-%m-%d %H:%M UTC')"
else
    clear_alert "scraper_down"
fi

# Scraper RSS too high
if [ "${SCRAPER_RSS_MB:-0}" -ge "$SCRAPER_RSS_WARN_MB" ]; then
    alert_once "scraper_rss" "<b>SCRAPER MEMORY HIGH: ${SCRAPER_RSS_MB}M</b>
Possible memory leak. Consider restarting.
Uptime: $(( SCRAPER_UPTIME_S / 3600 ))h"
else
    clear_alert "scraper_rss"
fi

# Load
LOAD_HIGH=$(echo "$LOAD_1M >= $LOAD_WARN" | bc -l 2>/dev/null || echo 0)
if [ "${LOAD_HIGH:-0}" -eq 1 ]; then
    alert_once "load_warn" "<b>LOAD HIGH: ${LOAD_1M}</b> (5m: ${LOAD_5M})
$(top -bn1 | head -12 | tail -5)"
else
    clear_alert "load_warn"
fi

# Journal bloat (>500MB)
if [ "${JOURNAL_MB:-0}" -ge 500 ]; then
    alert_once "journal_bloat" "<b>JOURNAL LOGS: ${JOURNAL_MB}M</b>
Run: journalctl --vacuum-size=200M"
else
    clear_alert "journal_bloat"
fi

# ── Push metrics to Supabase ──
if [ -n "${SUPABASE_URL:-}" ] && [ -n "${SUPABASE_SERVICE_ROLE_KEY:-}" ]; then
    curl -s -X POST "${SUPABASE_URL}/rest/v1/vps_metrics" \
        -H "apikey: ${SUPABASE_SERVICE_ROLE_KEY}" \
        -H "Authorization: Bearer ${SUPABASE_SERVICE_ROLE_KEY}" \
        -H "Content-Type: application/json" \
        -H "Prefer: return=minimal" \
        -d "{
            \"cpu_pct\": ${CPU_PCT},
            \"ram_used_mb\": ${RAM_USED},
            \"ram_total_mb\": ${RAM_TOTAL},
            \"ram_pct\": ${RAM_PCT},
            \"swap_used_mb\": ${SWAP_USED:-0},
            \"disk_used_gb\": ${DISK_USED_GB},
            \"disk_total_gb\": ${DISK_TOTAL_GB},
            \"disk_pct\": ${DISK_PCT},
            \"scraper_rss_mb\": ${SCRAPER_RSS_MB},
            \"scraper_cpu_pct\": ${SCRAPER_CPU},
            \"scraper_uptime_s\": ${SCRAPER_UPTIME_S},
            \"load_1m\": ${LOAD_1M},
            \"load_5m\": ${LOAD_5M},
            \"journal_size_mb\": ${JOURNAL_MB}
        }" > /dev/null 2>&1 || true
fi

# ── Auto-cleanup: purge old metrics (>30 days) once per day ──
CLEANUP_FLAG="${STATE_DIR}/last_cleanup"
if [ ! -f "$CLEANUP_FLAG" ] || [ $(( $(date +%s) - $(stat -c %Y "$CLEANUP_FLAG" 2>/dev/null || echo 0) )) -ge 86400 ]; then
    if [ -n "${SUPABASE_URL:-}" ] && [ -n "${SUPABASE_SERVICE_ROLE_KEY:-}" ]; then
        CUTOFF=$(date -u -d '30 days ago' '+%Y-%m-%dT%H:%M:%SZ')
        curl -s -X DELETE "${SUPABASE_URL}/rest/v1/vps_metrics?recorded_at=lt.${CUTOFF}" \
            -H "apikey: ${SUPABASE_SERVICE_ROLE_KEY}" \
            -H "Authorization: Bearer ${SUPABASE_SERVICE_ROLE_KEY}" \
            -H "Prefer: return=minimal" > /dev/null 2>&1 || true
        touch "$CLEANUP_FLAG"
    fi
fi
