from llm.tokenizer import tokenizer, trainer
import os
import json
def read_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            yield d['text']


def find_txt_files(root_dir):
    txt_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.txt'):
                txt_files.append(os.path.join(dirpath, filename))
    return txt_files


if __name__ == '__main__':
    book_data_dir = 'data/text/books'
    tokenizer_data_path = 'data/text/tokenizer_train.jsonl'
    tokenizer_path = 'checkpoints/tokenizer'

    txt_list = find_txt_files(book_data_dir)

    tokenizer.train_from_iterator(read_data(tokenizer_data_path), trainer)
    tokenizer.train(txt_list, trainer)

    os.makedirs(tokenizer_path, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_path, 'tokenizer.json'))
