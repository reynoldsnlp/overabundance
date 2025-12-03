import sys
import zipfile
from xml.etree import ElementTree as ET
import openpyxl

def parse_shared_strings(xlsx_path):
    # Extract sharedStrings.xml from the xlsx zip
    with zipfile.ZipFile(xlsx_path, 'r') as z:
        with z.open('xl/sharedStrings.xml') as f:
            tree = ET.parse(f)
            root = tree.getroot()
            ns = {'a': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
            shared_strings = {}
            for si in root.findall('a:si', ns):
                # If there are rich text runs
                runs = si.findall('a:r', ns)
                if runs:
                    s = ''
                    for r in runs:
                        t = r.find('a:t', ns)
                        text = t.text if t is not None else ''
                        rpr = r.find('a:rPr', ns)
                        is_bold = False
                        if rpr is not None and rpr.find('a:b', ns) is not None:
                            is_bold = True
                        if is_bold:
                            s += f'<b>{text}</b>'
                        else:
                            s += text
                    # Use the plain text as the key
                    plain = ''.join([r.find('a:t', ns).text if r.find('a:t', ns) is not None else '' for r in runs])
                    shared_strings[plain] = s
                else:
                    # Plain string
                    t = si.find('a:t', ns)
                    val = t.text if t is not None else ''
                    shared_strings[val] = val
            return shared_strings

def excel_to_tsv_with_bold(xlsx_path, tsv_path):
    shared_strings = parse_shared_strings(xlsx_path)
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb.active
    with open(tsv_path, 'w', encoding='utf-8') as out:
        for row in ws.iter_rows():
            values = []
            for cell in row:
                value = '' if cell.value is None else str(cell.value)
                # If the cell is a shared string, try to get the formatted version
                if cell.data_type == 's' and value in shared_strings:
                    values.append(shared_strings[value])
                else:
                    if cell.font and cell.font.bold:
                        value = f'<b>{value}</b>'
                    values.append(value)
            out.write('\t'.join(values) + '\n')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python excel_to_tsv_with_bold_xml.py input.xlsx output.tsv")
        sys.exit(1)
    excel_to_tsv_with_bold(sys.argv[1], sys.argv[2])
