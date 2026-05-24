import csv

cats = ['color_words','taste_words','sound_words','shape_words']
for c in cats:
    total = sum(1 for line in open(f'data/{c}.txt', encoding='utf-8') if line.strip())
    pw_all = list(csv.DictReader(open(f'data/outputs/{c}/per_word_dim_average_abs.csv', encoding='utf-8')))
    pos_all = list(csv.DictReader(open(f'data/outputs/{c}/positional_dim_average_abs.csv', encoding='utf-8')))

    pw_max = {r['dim']: int(r['appearance_count']) for r in pw_all if r['group'] == 'max'}
    pw_min = {r['dim']: int(r['appearance_count']) for r in pw_all if r['group'] == 'min'}
    pos_max = {r['dim']: (int(r['appearance_count']), float(r['mean_abs_value'])) for r in pos_all if r['group'] == 'max'}
    pos_min = {r['dim']: (int(r['appearance_count']), float(r['mean_abs_value'])) for r in pos_all if r['group'] == 'min'}

    print(f'\n=== {c} ({total} words) ===')
    print('  EMERGENT+ (pos>50%, pw<20%):')
    for dim, (pos_cnt, pos_mean) in pos_max.items():
        pw_cnt = pw_max.get(dim, 0)
        if pos_cnt > total * 0.5 and pw_cnt < total * 0.2:
            print(f'    +dim{dim}: pos={pos_cnt}/{total}  pw={pw_cnt}/{total}  pos_mean={pos_mean:.3f}')

    print('  EMERGENT- (pos>50%, pw<20%):')
    for dim, (pos_cnt, pos_mean) in pos_min.items():
        pw_cnt = pw_min.get(dim, 0)
        if pos_cnt > total * 0.5 and pw_cnt < total * 0.2:
            print(f'    -dim{dim}: pos={pos_cnt}/{total}  pw={pw_cnt}/{total}  pos_mean={pos_mean:.3f}')

    print('  SUPPRESSED+ (pw>70%, pos<30%):')
    for dim, pw_cnt in pw_max.items():
        pos_cnt = pos_max.get(dim, (0, 0))[0]
        if pw_cnt > total * 0.7 and pos_cnt < total * 0.3:
            print(f'    +dim{dim}: pw={pw_cnt}/{total}  pos={pos_cnt}/{total}')

    print('  SUPPRESSED- (pw>70%, pos<30%):')
    for dim, pw_cnt in pw_min.items():
        pos_cnt = pos_min.get(dim, (0, 0))[0]
        if pw_cnt > total * 0.7 and pos_cnt < total * 0.3:
            print(f'    -dim{dim}: pw={pw_cnt}/{total}  pos={pos_cnt}/{total}')
