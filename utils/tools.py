import os


def get_file_paths(folder_path):
    # 获取文件夹下所有文件的路径
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    print(len(file_paths))
    return file_paths


def get_file_paths_type(folder_path, file_types):
    if type(file_types) is not list:
        file_types = [file_types]
    # 获取文件夹下所有文件的路径
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if get_file_type(file) in file_types:
                file_paths.append(os.path.join(root, file))
    print(len(file_paths))
    return file_paths


def get_file_type(file_path):
    return file_path.split('.')[-1].lower()


def get_file_name(file_path):
    return file_path.split('/')[-1].split('.')[0]


def create_new_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def create_project_folder(project_folder, sub_folder):
    create_new_folder(project_folder)
    for f in sub_folder:
        new_folder = f'{project_folder}/{f}'
        create_new_folder(new_folder)


def type_cnt(path_list):
    cnt_info = {}
    for path in path_list:
        _type = get_file_type(path)
        if _type not in cnt_info:
            cnt_info[_type] = 1
        else:
            cnt_info[_type] += 1
    print(cnt_info)
