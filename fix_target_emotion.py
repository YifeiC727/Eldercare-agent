import json
import glob

files = sorted(glob.glob('output_batches/eldercare_15000_dialogues_part[1-9].jsonl') + glob.glob('output_batches/eldercare_15000_dialogues_part10.jsonl'))
count = 0
for fn in files:
    new_lines = []
    with open(fn, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            # 找到最后一轮user
            for i in range(len(d['dialogue'])-1, -1, -1):
                turn = d['dialogue'][i]
                if turn['role'] == 'user':
                    if 'emotion' in turn:
                        d['target_emotion'] = turn['emotion']
                        del turn['emotion']
                        count += 1
                    break
            new_lines.append(json.dumps(d, ensure_ascii=False))
    with open(fn, 'w', encoding='utf-8') as f:
        for line in new_lines:
            f.write(line + '\n')
print(f'已处理前10个output_batches文件，共{count}条数据已将最后一轮user的emotion字段转为target_emotion，并从dialogue中删除。')