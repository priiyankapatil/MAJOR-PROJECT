import json
import os

extracted_dir = 'data/extracted_text'

files_to_check = ['Apple.json', 'Crop-Management_1.json']

for fname in files_to_check:
    path = os.path.join(extracted_dir, fname)
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except UnicodeDecodeError:
            with open(path, encoding='latin-1') as f:
                data = json.load(f)
        
        total_chars = sum(len(p['text']) for p in data['pages'])
        print(f'{fname}:')
        print(f'  Pages: {data["total_pages"]}')
        print(f'  Total characters: {total_chars}')
        if total_chars > 0:
            sample = data["pages"][0]["text"][:100]
            print(f'  Sample text: {sample}...')
        else:
            print(f'  Sample text: [EMPTY - NO TEXT EXTRACTED]')
        print()
