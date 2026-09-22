#!/usr/bin/env bash
# watchdog_c4.sh — monitor for the C4 representation-isolation baselines
# (m4base_text_s1-3 + m4base_random_s1-3), launched 2026-09-02 ~12:23.
# Alerts Feishu on: process count change, new tracebacks, per-mode completion.
WEBHOOK="https://open.feishu.cn/open-apis/bot/v2/hook/REDACTED-WEBHOOK"
cd /root/autodl-tmp/genrl
send() { curl -s -X POST "$WEBHOOK" -H "Content-Type: application/json" \
  -d "{\"msg_type\":\"text\",\"content\":{\"text\":\"$1\"}}" >/dev/null; }
LAST_ERR=0; LAST_N=-1; DONE_T=0; DONE_R=0
send "[c4] watchdog 启动 $(date '+%F %T')"
while true; do
  N=$(pgrep -f "run_c4.sh" | wc -l)
  if [ "$N" != "$LAST_N" ]; then
    send "[c4] 进程数 $LAST_N -> $N ($(date '+%H:%M:%S'))"
    LAST_N=$N
  fi
  ERR=$(cat logs/m4base_text_s*.log logs/m4base_random_s*.log 2>/dev/null | grep -c Traceback)
  if [ "$ERR" != "$LAST_ERR" ]; then
    send "[c4] 累计 Traceback $LAST_ERR -> $ERR ($(date '+%H:%M:%S'))"
    LAST_ERR=$ERR
  fi
  if [ "$DONE_T" = 0 ] && grep -q "PARTITION_C_EXIT" logs/phase_C_c4_text.log 2>/dev/null; then
    send "[c4] text 3/3 完成 ($(date '+%H:%M:%S'))"; DONE_T=1
  fi
  if [ "$DONE_R" = 0 ] && grep -q "PARTITION_C_EXIT" logs/phase_C_c4_random.log 2>/dev/null; then
    send "[c4] random 3/3 完成 ($(date '+%H:%M:%S'))"; DONE_R=1
  fi
  if [ "$DONE_T" = 1 ] && [ "$DONE_R" = 1 ]; then
    send "[c4] 全部完成，可聚合/拉取 ($(date '+%H:%M:%S'))"; exit 0
  fi
  sleep 300
done
