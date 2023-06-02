import os
import re
import pdfplumber
import numpy as np
from langchain.document_loaders import UnstructuredPDFLoader
from pydantic import BaseModel, Extra, Field, root_validator


class Document(BaseModel):
    """Interface for interacting with a document."""

    page_content: str
    metadata: dict = Field(default_factory=dict)


def get_table_elemens_del(tables, page_number):
    first_cell = {}
    table_dict = {}
    for table_index in range(len(tables)):
        table_dict[f'table-{page_number}-{table_index}'] = tables[table_index]
        for row in tables[table_index]:
            for r in row:
                if r is not None:
                    if r != '' and os.linesep not in r:
                        first_cell[r] = f'table-{page_number}-{table_index}'
                        break
            for cell in row:
                if cell is not None:
                    if os.linesep in cell:
                        cell_lst = cell.split(os.linesep)
                        for c in cell_lst:
                            if c != '':
                                first_cell[c] = f'table-{page_number}-{table_index}'
    return first_cell, table_dict


def del_table_contents(text, tables, page_number):
    first_cell, table_dict = get_table_elemens_del(tables, page_number)
    new_text = []
    text_lst = text.split(os.linesep)
    first_cell_lst = tuple(first_cell.keys())
    if len(first_cell) > 0:
        for t in text_lst:
            if t.startswith(first_cell_lst):
                table_index = 'dsgfggsdgfdg'
                _key = t.split(' ')[0]
                for x in first_cell.keys():
                    if x.startswith(_key):
                        table_index = first_cell[x] + os.linesep
                        break
                if table_index == 'dsgfggsdgfdg':
                    new_text.append(t + os.linesep)
                elif table_index not in new_text:
                    new_text.append(table_index)
            else:
                new_text.append(t + os.linesep)

    return new_text, table_dict


def to_original_format_table(table):
    new_table = ""
    for row in table:
        row_text = " ".join(str(item).replace(os.linesep, '') for item in row).replace('None', '')
        new_table += row_text + "\n"
    return new_table


def new_table_dict(table_dict):
    new = {}
    for k in table_dict:
        new[k] = to_original_format_table(table_dict[k])
    return new


def to_original_format(content, table_dict):
    new_content = []
    for text in content:
        if text.strip(os.linesep) in table_dict:
            new_content.append(table_dict[text.strip(os.linesep)])
        else:
            _content = []
            for sentence in text.split(os.linesep):
                if sentence.strip(os.linesep).endswith(('。', '：', ':')):
                    _content.append(sentence + os.linesep)
                else:
                    if len(re.sub(r"[a-zA-Z0-9\s\W]+", "", sentence)) < 15 and len(sentence) > 5:
                        _content.append(sentence + os.linesep)
                    else:
                        _content.append(sentence.strip(os.linesep))
            new_content.append(''.join(_content))
    return ''.join(new_content).split(os.linesep)


def get_sub_docs(elements, metadata):
    text = "\n\n".join([el.strip(os.linesep) for el in elements])
    docs = [Document(page_content=text, metadata=metadata)]
    return docs


def pdf_parser(file_path):
    with pdfplumber.open(file_path) as pdf:
        extracted_content = []
        table_dict = {}
        for page in pdf.pages:
            text = page.extract_text()
            tables = page.extract_tables()

            if len(tables) > 0:
                text, _table_dict = del_table_contents(text, tables, page.page_number)
                extracted_content.extend(text)
                table_dict.update(_table_dict)
            else:
                extracted_content.extend(text.split(os.linesep))

    table_dict = new_table_dict(table_dict)
    metadata = {'source': file_path}
    elements = to_original_format(extracted_content, table_dict)
    if if_chinese_doc(elements) > 0.5:
        doc = get_sub_docs(elements, metadata)
    else:
        loader = UnstructuredPDFLoader(file_path)
        doc = loader.load()

    return doc


def has_chinese_characters(s):
    for c in s:
        if re.match('[\u4e00-\u9fa5\u3400-\u4DBF\U00020000-\U0002A6DF\U0002A700-\U0002B73F\U0002B740-\U0002B81F]', c):
            return True
    return False


def if_chinese_doc(elements):
    scores = []
    for e in elements:
        scores.append(has_chinese_characters(e))
    return np.mean(scores)


if __name__ == "__main__":
    path = 'xxx.pdf'
    elements = pdf_parser(path)
