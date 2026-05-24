import json

data = json.load(open('data/outputs/neuron_backup/layer_skip_results.json', encoding='utf-8'))
print(f"共 {len(data)} 层跳过实验\n")

# 排名条形图（排名越低越好 → 条越长）
print("  跳过层  rain排名     logit    排名可视化（条越长=排名越靠前）")
print("  " + "-"*70)
for r in data:
    s = r["skip_layer"]
    rank = r["rain_rank"]
    logit = r["rain_logit"]
    bar_len = max(1, int((130000 - rank) / 3000))
    bar = "█" * bar_len
    print(f"  层{s:2d}   #{rank:>6}   {logit:>7.3f}   {bar}")

print()
best  = min(data, key=lambda x: x["rain_rank"])
worst = max(data, key=lambda x: x["rain_rank"])
print(f"最佳 (rain排名最靠前): 跳过第{best['skip_layer']}层 -> rain #{best['rain_rank']}  logit={best['rain_logit']:.3f}")
print(f"最差 (rain排名最靠后): 跳过第{worst['skip_layer']}层 -> rain #{worst['rain_rank']}  logit={worst['rain_logit']:.3f}")

# 特殊层分析
print("\n  特殊层（排名异常好或差）:")
avg_rank = sum(r["rain_rank"] for r in data) / len(data)
print(f"  平均排名: #{avg_rank:.0f}")
good = [r for r in data if r["rain_rank"] < avg_rank * 0.5]
bad  = [r for r in data if r["rain_rank"] > avg_rank * 1.5]
print(f"  排名异常靠前的层: {[r['skip_layer'] for r in good]}")
print(f"  排名异常靠后的层: {[r['skip_layer'] for r in bad]}")

# top1词的变化
print("\n  各层Top-1词:")
prev_top1 = None
for r in data:
    top1 = r["top10"][0]["token"]
    mark = " <-- 变化!" if top1 != prev_top1 else ""
    print(f"  跳过层{r['skip_layer']:2d}: {repr(top1):<20} logit={r['top10'][0]['logit']:.3f}{mark}")
    prev_top1 = top1
