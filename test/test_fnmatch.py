import json
from api.config import DEFAULT_EXCLUDED_FILES
import fnmatch
from tqdm import tqdm
from collections import defaultdict

final_excluded_files = set(DEFAULT_EXCLUDED_FILES)
excluded_files = list(final_excluded_files)
dir_path = "/jyl/codehub/cdk"
result  = defaultdict(int)


from pathlib import Path
folder = Path(dir_path).resolve()
files = [f for f in folder.rglob("*") if f.is_file()]
for f in tqdm(files):
    file_name = f.name
    is_excluded_file_name = any(
                fnmatch.fnmatch(file_name, excluded_file) or file_name == excluded_file
                for excluded_file in excluded_files
            )
    if not is_excluded_file_name:
        result[f.suffix] += 1
result = dict(sorted(result.items(), key = lambda x:x[1],reverse=True))
print(result)
# with open("suffix_counts_hal.json","w",encoding='utf-8') as f:
#     json.dump(result,f,indent=2,ensure_ascii=False)

# import json

# expected = ['.h', '.c', '.lua', '', '.cpp', '.md', '.py', '.C']
# result = set({})

# with open("suffix_counts_hal.json","r",encoding='utf-8') as f:
#     result1 = json.load(f)

# with open("suffix_counts_cdk.json","r",encoding='utf-8') as f:
#     result2 = json.load(f)

# for key in result1:
#     if key not in expected:
#         result.add(f'*{key}')
# for key in result2:
#     if key not in expected:
#         result.add(f'*{key}')

# print(result)