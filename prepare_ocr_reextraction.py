import os
import json

# Files to re-extract with OCR
scanned_pdfs = [
    'Apple.pdf', 'Banana.pdf', 'Blackgram.pdf', 'Cardamom.pdf',
    'Chickpea.pdf', 'Citrus.pdf', 'Cotton.pdf', 'Ginger and Turmeric.pdf',
    'Grapes.pdf', 'mango.pdf', 'Millets.pdf', 'Mustard.pdf',
    'Pepper.pdf', 'Redgram.pdf', 'Soybean.pdf', 'Sugarcane.pdf', 'Sunflower.pdf'
]

print("Cleaning up for OCR re-extraction...\n")

# Delete old extraction files
for pdf in scanned_pdfs:
    json_file = f'data/extracted_text/{pdf.replace(".pdf", ".json")}'
    if os.path.exists(json_file):
        os.remove(json_file)
        print(f'✅ Deleted: {json_file}')

# Mark as not processed in registry
registry_path = 'data/metadata/registry.json'
if os.path.exists(registry_path):
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    removed_count = 0
    for pdf in scanned_pdfs:
        if pdf in registry['processed_files']:
            del registry['processed_files'][pdf]
            removed_count += 1
            print(f'✅ Removed from registry: {pdf}')
    
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f'\n✅ Registry updated ({removed_count} PDFs marked for re-extraction)')
