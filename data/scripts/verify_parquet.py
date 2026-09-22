import pandas as pd
import os
from config import ALL_CHUNKS_FILE, PARQUET_ENGINE

# Load the file
df = pd.read_parquet(ALL_CHUNKS_FILE, engine=PARQUET_ENGINE)

print('=== PARQUET FILE HEALTH CHECK ===')
print(f'Total chunks     : {len(df):,}')
print(f'Columns          : {list(df.columns)}')
print(f'File size        : {os.path.getsize(ALL_CHUNKS_FILE)/1024/1024:.2f} MB')
print()
print('=== SAMPLE CHUNK ===')
print(f'First chunk_id   : {df.iloc[0].chunk_id}')
print(f'Last chunk_id    : {df.iloc[-1].chunk_id}')
print(f'Sample text      : {df.iloc[0].text[:100]}...')
print()
print('=== SOURCE DISTRIBUTION ===')
print(df.source_file.value_counts().to_string())