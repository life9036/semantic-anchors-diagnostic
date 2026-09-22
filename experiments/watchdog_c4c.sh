#!/usr/bin/env bash
# watchdog_c4c.sh — monitor Cheetah C4 representation-isolation runs
# (c5base_text_s1-3 + c5base_random_s1-3). Alerts Feishu on process-count
# changes, new tracebacks, and per-partition completion.
WEBHOOK="https://open.feishu.cn/open-apis/bot/v2/hook/REDACTED-WEBHOOK"
cd /root/autodl-tmp/genrl
send() { curl -s -X POST "$WEBHOOK" -H "Content-Type: application/json" \
  -d "{\"msg_type\":\"text\",\"content\":{\"text\":\"$1\"}}" >/dev/null; }
LAST_ERR=0; LAST_N=-1; DONE_A=0; DONE_B=0
send "[c4c] watchdog 启动 $(date '+%F %T')"
while true; do
  N=$(pgrep -f "run_cheetah_c4.sh" | wc -l)
  if [ "$N" != "$LAST_N" ]; then
    send "[c4c] 进程数 $LAST_N -> $N ($(date '+%H:%M:%S'))"
    LAST_N=$N
  fi
  ERR=$(cat logs/c5base_text_s*.log logs/c5base_random_s*.log 2>/dev/null | grep -c Traceback)
  if [ "$ERR" != "$LAST_ERR" ]; then
    send "[c4c] 累计 Traceback $LAST_ERR -> $ERR ($(date '+%H:%M:%S'))"
    LAST_ERR=$ERR
  fi
  if [ "$DONE_A" = 0 ] && grep -q "PARTITION_C4_A_EXIT" logs/cheetah_c4_A.log 2>/dev/null; then
    send "[c4c] A 分区 4/4 完成 ($(date '+%H:%M:%S'))"; DONE_A=1
  fi
  if [ "$DONE_B" = 0 ] && grep -q "PARTITION_C4_B_EXIT" logs/cheetah_c4_B.log 2>/dev/null; then
    send "[c4c] B 分区 2/2 完成 ($(date '+%H:%M:%S'))"; DONE_B=1
  fi
  if [ "$DONE_A" = 1 ] && [ "$DONE_B" = 1 ]; then
    send "[c4c] 全部完成，可聚合/拉取 ($(date '+%H:%M:%S'))"; exit 0
  fi
  sleep 300
done
