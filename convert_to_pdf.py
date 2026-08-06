#!/usr/bin/env python3
"""
Convert PROJECT_OVERVIEW.md and QUERY_GATE_AND_TRUST_SCORING.md to PDF format
Using reportlab for PDF generation or HTML as fallback
"""

import os
import sys
import re
from pathlib import Path

def convert_markdown_to_pdf():
    """Convert markdown files to PDF using reportlab"""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
        from reportlab.lib.units import inch, pt
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Preformatted
        from reportlab.lib import colors
        
        md_files = [
            'PROJECT_OVERVIEW.md',
            'QUERY_GATE_AND_TRUST_SCORING.md'
        ]
        
        for md_file in md_files:
            if not os.path.exists(md_file):
                print(f"⚠️  {md_file} not found")
                continue
            
            print(f"📄 Converting {md_file}...")
            
            # Read markdown
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Generate PDF filename
            pdf_file = md_file.replace('.md', '.pdf')
            
            # Create PDF document
            doc = SimpleDocTemplate(pdf_file, pagesize=letter,
                                  rightMargin=0.75*inch, leftMargin=0.75*inch,
                                  topMargin=0.75*inch, bottomMargin=0.75*inch)
            
            # Get styles
            styles = getSampleStyleSheet()
            story = []
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#0066cc'),
                spaceAfter=12,
                alignment=TA_CENTER,
                borderPadding=10,
            )
            
            heading2_style = ParagraphStyle(
                'CustomHeading2',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#003d99'),
                spaceAfter=8,
                spaceBefore=12,
            )
            
            heading3_style = ParagraphStyle(
                'CustomHeading3',
                parent=styles['Heading3'],
                fontSize=12,
                textColor=colors.HexColor('#0052a3'),
                spaceAfter=6,
                spaceBefore=10,
            )
            
            normal_style = ParagraphStyle(
                'CustomNormal',
                parent=styles['Normal'],
                fontSize=11,
                alignment=TA_JUSTIFY,
                spaceAfter=6,
                leading=16,
            )
            
            code_style = ParagraphStyle(
                'CustomCode',
                parent=styles['Normal'],
                fontSize=9,
                fontName='Courier',
                backColor=colors.HexColor('#f4f4f4'),
                textColor=colors.HexColor('#333333'),
                spaceAfter=8,
                leftIndent=12,
                rightIndent=12,
            )
            
            # Parse markdown
            lines = content.split('\n')
            i = 0
            in_code_block = False
            code_buffer = []
            
            while i < len(lines):
                line = lines[i]
                
                # Code blocks
                if line.strip().startswith('```'):
                    if in_code_block:
                        # End code block
                        code_text = '\n'.join(code_buffer)
                        if code_text.strip():
                            story.append(Preformatted(code_text, code_style))
                            story.append(Spacer(1, 0.2*inch))
                        in_code_block = False
                        code_buffer = []
                    else:
                        # Start code block
                        in_code_block = True
                    i += 1
                    continue
                
                if in_code_block:
                    code_buffer.append(line)
                    i += 1
                    continue
                
                # H1 - Title
                if line.startswith('# '):
                    text = line[2:].strip()
                    story.append(Paragraph(text, title_style))
                    story.append(Spacer(1, 0.3*inch))
                    i += 1
                    continue
                
                # H2
                if line.startswith('## '):
                    text = line[3:].strip()
                    story.append(Paragraph(text, heading2_style))
                    i += 1
                    continue
                
                # H3
                if line.startswith('### '):
                    text = line[4:].strip()
                    story.append(Paragraph(text, heading3_style))
                    i += 1
                    continue
                
                # H4, H5, H6
                if line.startswith('#### ') or line.startswith('##### ') or line.startswith('###### '):
                    count = len(line) - len(line.lstrip('#'))
                    text = line[count:].strip()
                    story.append(Paragraph(f"<b>{text}</b>", normal_style))
                    i += 1
                    continue
                
                # Horizontal line / page break
                if line.strip() in ['---', '___', '***']:
                    story.append(PageBreak())
                    i += 1
                    continue
                
                # Empty line
                if not line.strip():
                    story.append(Spacer(1, 0.15*inch))
                    i += 1
                    continue
                
                # Regular paragraph
                if line.strip():
                    # Simple formatting: **bold** and *italic*
                    text = line.strip()
                    text = text.replace('**', '<b>').replace('__', '</b>')
                    text = text.replace('*', '<i>').replace('_', '</i>')
                    
                    # Fix mismatched tags
                    text = re.sub(r'<i>\s*<i>', '<i>', text)
                    text = re.sub(r'</i>\s*</i>', '</i>', text)
                    text = re.sub(r'<b>\s*<b>', '<b>', text)
                    text = re.sub(r'</b>\s*</b>', '</b>', text)
                    
                    story.append(Paragraph(text, normal_style))
                
                i += 1
            
            # Build PDF
            doc.build(story)
            print(f"✅ Created: {pdf_file}")
        
        return True
    
    except ImportError as e:
        print(f"❌ reportlab not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False


def convert_to_html():
    """Alternative: Convert markdown to HTML (can be opened in browser and printed as PDF)"""
    try:
        import markdown
        
        md_files = [
            'PROJECT_OVERVIEW.md',
            'QUERY_GATE_AND_TRUST_SCORING.md'
        ]
        
        html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #fff;
            padding: 40px;
            max-width: 1000px;
            margin: 0 auto;
        }}
        h1 {{ 
            color: #0066cc; 
            border-bottom: 3px solid #0066cc;
            padding-bottom: 12px;
            margin: 40px 0 20px 0;
            page-break-after: avoid;
            font-size: 2em;
        }}
        h2 {{ 
            color: #003d99;
            margin: 30px 0 15px 0;
            page-break-after: avoid;
            font-size: 1.6em;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 8px;
        }}
        h3 {{
            color: #0052a3;
            page-break-after: avoid;
            margin: 20px 0 10px 0;
            font-size: 1.3em;
        }}
        h4, h5, h6 {{
            page-break-after: avoid;
            margin: 15px 0 8px 0;
        }}
        p {{
            margin: 12px 0;
            text-align: justify;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', 'Monaco', monospace;
            font-size: 0.95em;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            border-left: 4px solid #0066cc;
            margin: 15px 0;
            line-height: 1.5;
            font-size: 0.9em;
        }}
        pre code {{
            background: none;
            padding: 0;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f0f0f0;
            font-weight: bold;
            color: #333;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        blockquote {{
            border-left: 4px solid #0066cc;
            margin: 15px 0;
            padding-left: 15px;
            color: #666;
            font-style: italic;
        }}
        a {{
            color: #0066cc;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        ul, ol {{
            margin: 15px 0 15px 25px;
        }}
        li {{
            margin: 8px 0;
        }}
        em, i {{
            font-style: italic;
        }}
        strong, b {{
            font-weight: bold;
            color: #000;
        }}
        @media print {{
            body {{ margin: 20px; }}
            h1 {{ page-break-after: avoid; }}
            h2 {{ page-break-after: avoid; }}
            h3 {{ page-break-after: avoid; }}
            pre {{ page-break-inside: avoid; }}
            table {{ page-break-inside: avoid; }}
        }}
    </style>
</head>
<body>
    {content}
</body>
</html>
"""
        
        for md_file in md_files:
            if not os.path.exists(md_file):
                print(f"⚠️  {md_file} not found")
                continue
            
            print(f"📄 Converting {md_file} to HTML...")
            
            # Read markdown
            with open(md_file, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            # Convert to HTML
            html_content = markdown.markdown(
                md_content,
                extensions=['extra', 'codehilite', 'toc', 'tables']
            )
            
            # Generate filename
            html_file = md_file.replace('.md', '.html')
            title = os.path.basename(md_file)
            
            # Write HTML file
            full_html = html_template.format(
                title=title,
                content=html_content
            )
            
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(full_html)
            
            print(f"✅ Created: {html_file}")
            print(f"   📌 Open in browser and press Ctrl+P to print as PDF")
        
        return True
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Main conversion function"""
    print("\n" + "="*60)
    print("  📄 Markdown to PDF Converter")
    print("="*60 + "\n")
    
    # Try reportlab first
    print("🔍 Trying reportlab...")
    if convert_markdown_to_pdf():
        print(f"\n✅ Conversion successful!")
        return 0
    
    # Fallback to HTML
    print("\n🔍 Trying HTML conversion (alternative)...")
    if convert_to_html():
        print(f"\n✅ HTML files created!")
        print("   Open in your browser and use Ctrl+P to save as PDF")
        return 0
    
    print("\n❌ All conversion methods failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
