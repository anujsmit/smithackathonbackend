import io
import pdfplumber

def extract_text_from_pdf_fileobj(fileobj):
    """
    Extract text from PDF file object
    """
    text = ""
    try:
        with pdfplumber.open(fileobj) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        raise Exception(f"PDF extraction failed: {str(e)}")
    
    return text

def extract_text_from_txt_fileobj(fileobj, encoding='utf-8'):
    """
    Extract text from TXT file object
    """
    try:
        # Reset file pointer to beginning
        fileobj.seek(0)
        data = fileobj.read()
        
        if isinstance(data, bytes):
            data = data.decode(encoding, errors='ignore')
        
        return data
    except Exception as e:
        raise Exception(f"Text extraction failed: {str(e)}")