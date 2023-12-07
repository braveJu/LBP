import os

data_src = 'C:/Users/ABC/Desktop/dongju/vision/Splited'
train_src = os.path.join(data_src, 'train')
valid_src = os.path.join(data_src, 'valid')


def get_train_info():
    f = open('train.txt', 'wt', encoding='utf-8')

    # 트레인 폴더로 이동
    os.chdir(train_src)
    train_data_dir = os.listdir(train_src)[:11]
    lines = []
    for label, d in enumerate(train_data_dir):
        data_dir = os.path.join(train_src, d)
        file_list = os.listdir(data_dir)
        for file in file_list:
            absol_file_root = os.path.join(data_dir, file)
            line = f"{absol_file_root},{label}\n".replace('\\', '/')
            lines.append(line)
    f.writelines(lines)
    f.close()


def get_valid_info():
    f = open('valid.txt', 'wt', encoding='utf-8')

    # 트레인 폴더로 이동
    os.chdir(valid_src)
    valid_data_dir = os.listdir(valid_src)[:11]
    lines = []
    for label, d in enumerate(valid_data_dir):
        data_dir = os.path.join(valid_src, d)
        file_list = os.listdir(data_dir)
        for file in file_list:
            absol_file_root = os.path.join(data_dir, file)
            line = f"{absol_file_root},{label}\n".replace('\\', '/')
            lines.append(line)
    f.writelines(lines)
    f.close()


if __name__ == '__main__':
    # get_train_info()
    get_valid_info()
