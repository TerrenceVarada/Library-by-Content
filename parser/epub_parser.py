from langchain.document_loaders import UnstructuredEPubLoader


def epub_parser(path):
    try:
        loader = UnstructuredEPubLoader(path)
        doc = loader.load()
    except:
        print('error ', path)
        doc = []
    return doc


if __name__ == "__main__":
    path = 'xxx.pdf'
    elements = epub_parser(path)
