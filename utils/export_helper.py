import io

def generate_txt(text: str) -> bytes:
    return text.encode('utf-8')

def generate_docx(text: str) -> bytes:
    try:
        import docx
        doc = docx.Document()
        doc.add_paragraph(text)
        file_stream = io.BytesIO()
        doc.save(file_stream)
        return file_stream.getvalue()
    except Exception as e:
        return f"Error generating DOCX: {e}".encode('utf-8')

def generate_pdf(text: str) -> bytes:
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica", size=12)
        # Handle multi-line strings with explicit left-margin line advance
        for line in text.split('\n'):
            safe_line = line.encode('latin-1', 'replace').decode('latin-1')
            try:
                pdf.multi_cell(0, 10, text=safe_line, new_x="LMARGIN", new_y="NEXT")
            except TypeError:
                try:
                    pdf.multi_cell(0, 10, txt=safe_line, ln=1)
                except Exception:
                    pdf.multi_cell(0, 10, txt=safe_line)
                    pdf.ln()
                
        out = pdf.output()
        if isinstance(out, (bytearray, bytes)):
            return bytes(out)
        else:
            return out.encode('latin-1')
    except Exception as e:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica", size=10)
        safe_msg = f"Document Export:\n{text}".encode('latin-1', 'replace').decode('latin-1')
        try:
            pdf.multi_cell(0, 10, text=safe_msg, new_x="LMARGIN", new_y="NEXT")
        except TypeError:
            pdf.multi_cell(0, 10, txt=safe_msg)
        out = pdf.output()
        return bytes(out) if isinstance(out, (bytearray, bytes)) else out.encode('latin-1')
