import re
import os

for path in ['ui/agrirag_ui.html', 'agrirag_ui.html']:
    if not os.path.exists(path):
        continue
    with open(path, 'r', encoding='utf-8') as f:
        html = f.read()

    # 1. Lighten the gradient overlay drastically so the sunrise, mountains & fields are clearly visible
    old_grad = """    .hero-section {
      width: 100%;
      background:
        linear-gradient(
          180deg, 
          rgba(13, 35, 25, 0.70) 0%, 
          rgba(20, 50, 36, 0.74) 45%, 
          rgba(27, 67, 50, 0.88) 80%,
          rgba(27, 67, 50, 0.96) 100%
        ),"""

    new_grad = """    .hero-section {
      width: 100%;
      background:
        linear-gradient(
          180deg, 
          rgba(10, 25, 18, 0.22) 0%, 
          rgba(12, 32, 22, 0.32) 42%, 
          rgba(18, 48, 33, 0.62) 80%,
          rgba(25, 62, 45, 0.86) 100%
        ),"""

    if old_grad in html:
        html = html.replace(old_grad, new_grad)
    else:
        # regex replace if whitespace differs
        html = re.sub(
            r'linear-gradient\(\s*180deg,\s*rgba\(13,\s*35,\s*25,\s*0\.70\)\s*0%,[\s\S]*?rgba\(27,\s*67,\s*50,\s*0\.96\)\s*100%\s*\)',
            '''linear-gradient(
          180deg, 
          rgba(10, 25, 18, 0.22) 0%, 
          rgba(12, 32, 22, 0.32) 42%, 
          rgba(18, 48, 33, 0.62) 80%,
          rgba(25, 62, 45, 0.86) 100%
        )''',
            html
        )

    # 2. Fix the title so "Growing Better Answers" glows in vibrant golden morning light
    old_title_span = """    .hero-title span {
      background: linear-gradient(90deg, #B7E4C7, #D8F3DC);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      filter: drop-shadow(0 2px 8px rgba(0, 0, 0, 0.2));
    }"""

    new_title_span = """    .hero-title span {
      color: #FFE680;
      background: linear-gradient(90deg, #FFF1A8, #C7F9CC);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      filter: drop-shadow(0 2px 12px rgba(0, 0, 0, 0.85));
    }"""

    if old_title_span in html:
        html = html.replace(old_title_span, new_title_span)

    # 3. Enhance text readability with strong dark text-shadows
    old_title = """    .hero-title {
      font-size: clamp(2.2rem, 5vw, 3.4rem);
      font-weight: 800;
      letter-spacing: -0.035em;
      line-height: 1.15;
      margin-bottom: 14px;
      color: #FFFFFF;
      text-shadow: 0 2px 14px rgba(0, 0, 0, 0.35), 0 1px 3px rgba(0, 0, 0, 0.5);
    }"""

    new_title = """    .hero-title {
      font-size: clamp(2.3rem, 5.2vw, 3.6rem);
      font-weight: 800;
      letter-spacing: -0.035em;
      line-height: 1.15;
      margin-bottom: 14px;
      color: #FFFFFF;
      text-shadow: 0 3px 20px rgba(0, 0, 0, 0.85), 0 1px 4px rgba(0, 0, 0, 0.95);
    }"""

    if old_title in html:
        html = html.replace(old_title, new_title)

    # 4. Make the 3 cards more translucent frosted glass so the green paddy is clearly visible behind them
    old_card = """    .hero-feature-card {
      background: rgba(18, 48, 35, 0.55);
      border: 1px solid rgba(255, 255, 255, 0.22);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border-radius: var(--radius);
      padding: 18px 16px;
      display: flex;
      flex-direction: column;
      align-items: center;
      text-align: center;
      gap: 8px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
      transition: var(--transition);
    }"""

    new_card = """    .hero-feature-card {
      background: rgba(12, 38, 25, 0.42);
      border: 1px solid rgba(255, 255, 255, 0.32);
      backdrop-filter: blur(8px);
      -webkit-backdrop-filter: blur(8px);
      border-radius: var(--radius);
      padding: 18px 16px;
      display: flex;
      flex-direction: column;
      align-items: center;
      text-align: center;
      gap: 8px;
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
      transition: var(--transition);
    }"""

    if old_card in html:
        html = html.replace(old_card, new_card)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Successfully brightened hero section in {path}!')
