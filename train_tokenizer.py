from llm.tokenizer import tokenizer, trainer
import os
import json
def read_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        total_l = len(lines)
        for i, l in enumerate(lines):
            d = json.loads(l)
            print(f'{i}/{total_l}')
            yield d['text']




if __name__ == '__main__':
    tokenizer_data_path = 'data/text/tokenizer_train.jsonl'
    tokenizer_path = 'checkpoints/tokenizer'


    tokenizer.train_from_iterator(read_data(tokenizer_data_path), trainer)

    os.makedirs(tokenizer_path, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_path, 'tokenizer.json'))
