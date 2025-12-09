import pickle

txt_path = '/mnt/petrelfs/xukaiyi/CodeSpace/rankcast/datasets/hko7/hko7_12min_list/hko7_rainy_valid.txt'

with open(txt_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

lines_list = [line.strip() for line in lines]


with open('/mnt/petrelfs/xukaiyi/CodeSpace/rankcast/datasets/hko7/hko7_12min_list/hko7_rainy_valid.pkl', 'wb') as pkl_file:
    pickle.dump(lines_list, pkl_file)
