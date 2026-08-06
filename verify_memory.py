import tracemalloc
import json
import pandas as pd
import os
from config import ALL_CHUNKS_FILE, PARQUET_ENGINE

# 1. Test JSON memory (using the backup file)
json_file = ALL_CHUNKS_FILE.replace('.parquet', '.json.backup')

print('--- Memory Usage Comparison ---')

if os.path.exists(json_file):
    tracemalloc.start()
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    _, json_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f'JSON peak memory   : {json_peak/1024/1024:.2f} MB')
else:
    print('JSON backup not found - skipping JSON test')

# 2. Test Parquet memory
tracemalloc.start()
df = pd.read_parquet(ALL_CHUNKS_FILE, engine=PARQUET_ENGINE)
_, parquet_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f'Parquet peak memory: {parquet_peak/1024/1024:.2f} MB')

# 3. File sizes
parquet_size = os.path.getsize(ALL_CHUNKS_FILE)/1024/1024
print(f'Parquet file size  : {parquet_size:.2f} MB')

if os.path.exists(json_file):
    json_size = os.path.getsize(json_file)/1024/1024
    print(f'JSON file size     : {json_size:.2f} MB')