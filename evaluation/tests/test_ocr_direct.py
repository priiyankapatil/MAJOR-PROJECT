import fitz
import easyocr
from PIL import Image
import io
import numpy as np

# Initialize OCR
print("Initializing OCR reader...")
reader = easyocr.Reader(['en'])

# Test on Apple.pdf
print("\nTesting OCR on Apple.pdf...")
doc = fitz.open('data/raw pdf/Apple.pdf')

# Check first page
page = doc[0]
text = page.get_text()  # Try normal text extraction
print(f"Normal text extraction: {len(text)} chars")

if len(text) == 0:
    print("No text found, trying OCR on image...")
    
    # Render page as image
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    print(f"Page rendered as image: {pix.width}x{pix.height}")
    
    # Convert to PIL Image
    img_data = pix.tobytes("ppm")
    img = Image.open(io.BytesIO(img_data))
    img_array = np.array(img)
    print(f"Image array shape: {img_array.shape}")
    
    # Run OCR
    print("Running OCR (this may take a moment)...")
    results = reader.readtext(img_array)
    print(f"OCR found {len(results)} text regions")
    
    if results:
        ocr_text = "\n".join([text for (bbox, text, conf) in results])
        print(f"OCR extracted: {len(ocr_text)} chars")
        print(f"\nSample text (first 300 chars):")
        print(ocr_text[:300])
    else:
        print("OCR found no text")

doc.close()
