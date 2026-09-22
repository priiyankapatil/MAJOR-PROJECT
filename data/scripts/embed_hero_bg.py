import base64
import os

with open('ui/hero_bg.jpg', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode('utf-8')

data_uri = f'data:image/jpeg;base64,{b64}'

for path in ['ui/agrirag_ui.html', 'agrirag_ui.html']:
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        updated = content.replace("url('hero_bg.jpg')", f"url('{data_uri}')")
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(updated)
        print(f'Successfully embedded base64 hero image in {path}!')
