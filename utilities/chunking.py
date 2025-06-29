from PyPDF2 import PdfReader
import io

def chunk_pdf_and_upload(bytes_data):
    pdf_stream = io.BytesIO(bytes_data)
    pdf_reader = PdfReader(pdf_stream)
    num_pages = len(pdf_reader.pages)

    chunks = []
    for i in range(num_pages):
        page_i = pdf_reader.pages[i]
        chunk_i = page_i.extract_text()
        chunks.append(chunk_i)

    return chunks