with open('/mnt/petrelfs/xukaiyi/CodeSpace/rankcast/datasets/hko7/hko7_list/hko7_rainy_valid.txt', 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f]

# 每隔两行取一个
new_lines = lines[::2]

with open('/mnt/petrelfs/xukaiyi/CodeSpace/rankcast/datasets/hko7/hko7_12min_list/hko7_rainy_valid.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(new_lines))
