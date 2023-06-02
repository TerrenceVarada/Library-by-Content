import re
import json
import functools
import shutil
import multiprocessing
import numpy as np
import pandas as pd
from config import *
from utils.tools import *
from tqdm import tqdm
from parser.pdf_parser import pdf_parser
from parser.epub_parser import epub_parser
from langchain.llms import LlamaCpp
from langchain.embeddings.llamacpp import LlamaCppEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_doc_info(path, output_folder):
    try:
        file_name = path.split('/')[-1].split('.')[0]
        if get_file_type(path) == 'epub':
            doc = epub_parser(path)
        elif get_file_type(path) == 'pdf':
            doc = pdf_parser(path)
        if doc:
            if len(doc[0].page_content) > 20:
                info = {'content': doc[0].page_content, 'metadata': doc[0].metadata}
                with open(f"{output_folder}/{file_name}.json", "w", encoding='utf-8') as f:
                    json.dump(info, f, ensure_ascii=False)
    except Exception as e:
        print(path, e)


def parse_files(path_list, output_folder, cpu_num=4):
    if cpu_num == -1:
        cpu_num = multiprocessing.cpu_count()
    cpu_num = min(cpu_num, multiprocessing.cpu_count())
    with multiprocessing.Pool(processes=cpu_num) as pool:
        pool.map(functools.partial(get_doc_info, output_folder=output_folder), path_list)


def run_new_files(input_files, output_files, output_folder, file_type):
    new = []
    for path in input_files:
        new_path = f'{output_folder}/{get_file_name(path)}.{file_type}'
        if new_path not in output_files:
            new.append(path)
    print(len(new))
    return new


def get_text(project_folder, files_folder, file_types):
    create_new_folder(project_folder)
    txt_folder = f'{project_folder}/txt'
    create_new_folder(txt_folder)

    path_list = get_file_paths_type(files_folder, file_types)
    txt_path_list = get_file_paths_type(txt_folder, file_types='json')
    path_list = run_new_files(path_list, txt_path_list, txt_folder, file_type='json')

    parse_files(path_list, txt_folder, cpu_num=10)
    return txt_folder


def exclude_length(s):
    # 将数字、中英文全半角括号替换为空字符串
    s = re.sub('[0-9（）()【】\[\]<>《》{}｛｝.“”‘’\'\"]+', '', s)
    # 计算字符串长度
    return len(s)


def get_new_content(content):
    new_content = []
    for c in content:
        if exclude_length(c) > 3:
            new_content.append(c)
    return new_content[:200]


def load_doc(path):
    with open(path, "r") as f:
        data = json.load(f)
        content = data['content'].split('\n\n')
        new_content = get_new_content(content)
        metadata = data['metadata']

    text = os.linesep.join(new_content)
    text = text.replace('\t', ' ')
    # num_tokens = llm.get_num_tokens(text)
    # logging.info(f"This doc has {num_tokens} tokens in it")
    return text, metadata


def get_summaries_embedding(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)

    docs = text_splitter.create_documents([text])

    print('Page Vectorize', len(docs))
    vectors = embeddings.embed_documents([x.page_content for x in docs])
    vectors = np.array([np.array(x) for x in vectors])

    return vectors.mean(axis=0)


def get_embedding(project_folder, txt_folder):
    vector_folder = f'{project_folder}/vector'
    create_new_folder(vector_folder)

    path_list = get_file_paths_type(txt_folder, file_types=['json'])
    output_path_list = get_file_paths_type(vector_folder, file_types=['json'])

    for i in tqdm(range(len(path_list))):
        path = path_list[i]
        file_name = get_file_name(path)
        write_path = f"{vector_folder}/{file_name}.json"
        if write_path not in output_path_list:
            text, metadata = load_doc(path)
            vectors = get_summaries_embedding(text)
            info = {'vectors': vectors, 'metadata': metadata}
            with open(write_path, "w") as f:
                f.write(json.dumps(info, ensure_ascii=False, default=NumpyEncoder))
    return vector_folder


def load_vector(path):
    with open(path, "r") as f:
        data = json.load(f)
        _vector = data['vectors'][0]
        _metadata = data['metadata']
    return _vector, _metadata


def cluster_vectors(vectors, num_clusters):
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto').fit(vectors)
    labels = kmeans.fit_predict(vectors)
    return labels, kmeans


# 定义聚类算法模型，并使用轮廓系数来评估聚类质量
def get_best_k(vectors, max_cluster, min_cluster):
    scores = []
    labels_info = {}
    for k in tqdm(range(min_cluster, max_cluster)):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto', max_iter=700)
        labels = kmeans.fit_predict(vectors)
        labels_info[k] = labels
        score = silhouette_score(vectors, labels)
        scores.append(score)
    best_k = np.argmax(scores) + 2  # 最优聚类数为轮廓系数最大的索引加2
    logging.info(f'Best k: {best_k}')
    return labels_info[best_k], cluster_vectors(vectors, best_k)


def get_closest_indices(num_clusters, vectors, kmeans):
    closest_indices = []

    for i in range(num_clusters):
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
        closest_index = np.argmin(distances)
        closest_indices.append(closest_index)
    selected_indices = sorted(closest_indices)
    logging.info(len(selected_indices))
    return selected_indices


def combine_lists_to_dict(keys, values):
    dict_result = {}
    for i in range(len(keys)):
        dict_result[keys[i]] = values[i]
    return dict_result


def move_files(res, selected_indices, out_folder):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    res = res[['metadata', 'labels']]
    selected_df = res.loc[selected_indices].sort_values(['labels'])
    selected_df['metadata'] = [get_file_name(x) for x in selected_df['metadata'].tolist()]
    selected_info = combine_lists_to_dict(selected_df['labels'].tolist(), selected_df['metadata'].tolist())
    for l in res['labels'].unique():
        _df = res[res['labels'] == l]
        try:
            sub_filder = f'{out_folder}/{l}-{get_file_name(selected_info[l])}'
        except:
            sub_filder = f'{out_folder}/{l}'
        if not os.path.exists(sub_filder):
            os.makedirs(sub_filder)
        for p in _df['metadata'].tolist():
            destination_file_path = f"{sub_filder}/{p.split('/')[-1]}"
            try:
                shutil.copy(p, destination_file_path)
            except:
                print(p, destination_file_path)


def get_vector_metadata(vector_folder):
    vector_lst = []
    metadata_lst = []
    path_list = get_file_paths_type(vector_folder, file_types=['json'])
    for i in tqdm(range(len(path_list))):
        path = path_list[i]
        _vector, _metadata = load_vector(path)
        vector_lst.append(_vector)
        metadata_lst.append(_metadata['source'])
    return metadata_lst, vector_lst


def cluster_embedding(vector_folder):
    metadata_lst, vector_lst = get_vector_metadata(vector_folder)
    if cluster_search:
        labels, kmeans = get_best_k(vector_lst, max_cluster=num_clusters, min_cluster=10)
    else:
        labels, kmeans = cluster_vectors(vector_lst, num_clusters=num_clusters)
    res = pd.DataFrame({'metadata': metadata_lst, 'labels': labels.tolist()})
    res.to_excel('res.xlsx')
    selected_indices = get_closest_indices(num_clusters, vector_lst, kmeans)
    print(selected_indices)
    move_files(res, selected_indices, out_folder='/Users/chenshi/Desktop/Library-New')


project_folder = 'Data'
cluster_search = False
num_clusters = 30

if __name__ == "__main__":
    txt_folder = get_text(project_folder, files_folder, file_types)
    llm = LlamaCpp(model_path=model_path)
    embeddings = LlamaCppEmbeddings(model_path=model_path, use_mlock=True)
    vector_folder = get_embedding(project_folder, txt_folder)
    cluster_embedding(vector_folder)
