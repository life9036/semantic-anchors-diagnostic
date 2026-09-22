#!/usr/bin/env bash
# Remote watchdog: monitors the Walker matrix and pushes Feishu notifications
# on key events (phase done / failure / all gates done). Runs as a setsid
# daemon on the remote (tmux is killed by this platform). Zero-impact when
# ~/.claude/feishu.json is absent (logs locally only).
# Usage: setsid bash experiments/watchdog_feishu.sh </dev/null >> logs/watchdog.log 2>&1 &
set -u
REPO=/root/autodl-tmp/genrl
cd "$REPO"
INTERVAL=300   # 5 min

notify() {  # notify <title> <color> <body-md>
  local cfg="$HOME/.claude/feishu.json"
  if [ ! -f "$cfg" ]; then
    echo "[$(date +%T)] (no feishu config) $1 :: $3" >> "$REPO/logs/watchdog.log"
    return
  fi
  python3 - "$cfg" "$1" "$2" "$3" <<'PYEOF'
import json, sys, urllib.request
cfg, title, color, body = sys.argv[1:5]
try:
    url = json.load(open(cfg)).get('webhook_url', '')
except Exception as e:
    print('cfg error:', e)
    sys.exit(0)
if not url:
    sys.exit(0)
payload = {
    "msg_type": "interactive",
    "card": {
        "header": {"title": {"tag": "plain_text", "content": title},
                   "template": color},
        "elements": [{"tag": "markdown", "content": body}],
    },
}
req = urllib.request.Request(
    url, data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"})
try:
    urllib.request.urlopen(req, timeout=10)
    print('feishu sent')
except Exception as e:
    print('feishu send failed:', e)
PYEOF
  echo "[$(date +%T)] feishu: $1" >> "$REPO/logs/watchdog.log"
}

declare -A sent
FLAG_BASE="$REPO/exp/main"

while true; do
  # failure
  if [ -f "$FLAG_BASE/PARTITION_FAILURE.flag" ] && [ -z "${sent[fail]:-}" ]; then
    sent[fail]=1
    tailA=$(tail -3 "$REPO/logs/phase_A.log" 2>/dev/null)
    tailB=$(tail -3 "$REPO/logs/phase_B.log" 2>/dev/null)
    notify "⚠️ 实验矩阵故障" "red" "**时间**: $(date '+%F %T')
**A**: $tailA
**B**: $tailB"
  fi
  # phase completions
  for ph in m1_A m1_B m2_A m2_B m3_A m3_B; do
    if [ -f "$FLAG_BASE/${ph}_DONE.flag" ] && [ -z "${sent[$ph]:-}" ]; then
      sent[$ph]=1
      notify "✅ 阶段完成: $ph" "green" "**时间**: $(date '+%F %T')"
    fi
  done
  # all gates done
  if [ -f "$FLAG_BASE/ALL_GATES_DONE.flag" ] && [ -z "${sent[gates]:-}" ]; then
    sent[gates]=1
    agg2=$(python3 -c "import json;d=json.load(open('$REPO/exp/main_results/aggregate_m2.json'));print('B2 描述子12/16=%s  ID≥25%%=%s  DIAYN_CI=%s'%(d['gate_descriptor_12of16'],d['gate_id_25pct'],d['gate_diayn_ci']))" 2>/dev/null || echo "aggregate_m2 解析失败")
    agg34=$(python3 -c "import json;d=json.load(open('$REPO/exp/main_results/aggregate_m34.json'));g=d['gates_b4'];print('B3 on>off=%s off失败单调=%s | B4 归因%s 插值%s 单调%s 覆盖%s'%(d['gate_m3_mixup_on_beats_off'],d['gate_m3_off_fails_monotone'],g['attribution'],g['interpolation'],g['monotone'],g['coverage']))" 2>/dev/null || echo "aggregate_m34 解析失败")
    notify "🏁 全部闸门聚合完成" "blue" "**时间**: $(date '+%F %T')
$agg2
$agg34"
  fi
  # anomaly: drivers gone, no failure flag, matrix clearly started
  n_drivers=$(pgrep -fc "run_matrix.sh" || true)
  if [ "$n_drivers" = "0" ] && [ ! -f "$FLAG_BASE/PARTITION_FAILURE.flag" ] \
     && [ ! -f "$FLAG_BASE/ALL_GATES_DONE.flag" ] \
     && { [ -f "$FLAG_BASE/m3_A_DONE.flag" ] || [ -f "$FLAG_BASE/m3_B_DONE.flag" ]; } \
     && [ -z "${sent[dead]:-}" ]; then
    sent[dead]=1
    notify "❓ 矩阵进程消失(非故障标记)" "orange" "**时间**: $(date '+%F %T')"
  fi
  sleep $INTERVAL
done
