import json
data = json.load(open('data/outputs/neuron_backup/head_ablation_results.json', encoding='utf-8'))
print(f'基线 rain排名=#{data["baseline_rain_rank"]}')
print('基线 Top-10:')
for i, t in enumerate(data['baseline_top10'], 1):
    print(f'  #{i:2d}  {repr(t["token"]):<20}  logit={t["logit"]:.4f}')
print()
for head_idx in [8, 9, 18, 30]:
    r = data['ablations'][head_idx]
    print(f'【关闭头{r["head"]:2d}】 rain#{r["rain_rank"]}  logit={r["rain_logit"]:.3f}')
    for i, t in enumerate(r['top10'], 1):
        print(f'  #{i:2d}  {repr(t["token"]):<20}  logit={t["logit"]:.4f}')
    print()
