#!/usr/bin/env bash
# watchdog_reeval.sh — minimal monitor for the R1 re-evaluation (2026-09-02).
# Alerts Feishu on: process count change, new tracebacks, partition completion.
WEBHOOK="https://open.feishu.cn/open-apis/bot/v2/hook/REDACTED-WEBHOOK"
cd /root/autodl-tmp/genrl
send() { curl -s -X POST "$WEBHOOK" -H "Content-Type: application/json" \
  -d "{\"msg_type\":\"text\",\"content\":{\"text\":\"$1\"}}" >/dev/null; }
LAST_ERR=0; LAST_N=-1; DONE_A=0; DONE_B=0
send "[reeval] watchdog 启动 $(date '+%F %T')"
while true; do
  N=$(pgrep -f "reeval_all.sh" | wc -l)
  if [ "$N" != "$LAST_N" ]; then
    send "[reeval] 进程数 $LAST_N -> $N ($(date '+%H:%M:%S'))"
    LAST_N=$N
  fi
  ERR=$(cat logs/phase_A_reeval.log logs/phase_B_reeval.log 2>/dev/null | grep -c Traceback)
  if [ "$ERR" != "$LAST_ERR" ]; then
    send "[reeval] 累计 Traceback $LAST_ERR -> $ERR ($(date '+%H:%M:%S'))"
    LAST_ERR=$ERR
  fi
  if [ "$DONE_A" = 0 ] && grep -q "PARTITION_A_EXIT" logs/phase_A_reeval.log 2>/dev/null; then
    send "[reeval] 分区 A 完成 ($(date '+%H:%M:%S'))"; DONE_A=1
  fi
  if [ "$DONE_B" = 0 ] && grep -q "PARTITION_B_EXIT" logs/phase_B_reeval.log 2>/dev/null; then
    send "[reeval] 分区 B 完成 ($(date '+%H:%M:%S'))"; DONE_B=1
  fi
  if [ "$DONE_A" = 1 ] && [ "$DONE_B" = 1 ]; then
    send "[reeval] 两分区全部完成，可接 C4 基线 ($(date '+%H:%M:%S'))"; exit 0
  fi
  sleep 300
done
