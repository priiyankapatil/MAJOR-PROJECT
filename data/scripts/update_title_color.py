import os
import re

for path in ['ui/agrirag_ui.html', 'agrirag_ui.html']:
    if not os.path.exists(path):
        continue
    with open(path, 'r', encoding='utf-8') as f:
        html = f.read()

    # Replace .hero-title and .hero-title span
    old_css = re.search(r'\.hero-title\s*\{[\s\S]*?\.hero-title span\s*\{[\s\S]*?\}', html)
    if old_css:
        new_css = """.hero-title {
      font-size: clamp(2.3rem, 5.2vw, 3.6rem);
      font-weight: 800;
      letter-spacing: -0.035em;
      line-height: 1.18;
      margin-bottom: 14px;
      color: #FFFFFF;
      text-shadow: 0 2px 8px rgba(0, 0, 0, 0.6);
    }

    .hero-title span {
      color: #143D2B; /* Elegant Deep Dark Green */
      background: none;
      -webkit-text-fill-color: #143D2B;
      text-shadow: 0 1px 3px rgba(255, 255, 255, 0.7), 0 0 15px rgba(255, 255, 255, 0.5);
      filter: none;
    }"""
        html = html[:old_css.start()] + new_css + html[old_css.end():]
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"Updated dark green title in {path}")
