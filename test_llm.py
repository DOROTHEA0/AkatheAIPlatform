from llm.config import AkatheV1Config
from llm.model import *
from llm.utils import count_parameters, generate_att_mask
import torch
from torch import optim
from transformers import PreTrainedTokenizerFast




if __name__ == '__main__':
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="checkpoints/tokenizer/tokenizer.json")
    text = '''您想要测试的新文本内容'''
    encoded = tokenizer(text)
    l = len(encoded["input_ids"])
    x = encoded["input_ids"][0: l - 1]
    y = encoded["input_ids"][1: l]

    x = torch.tensor(x, dtype=torch.int64).unsqueeze(0).cuda()
    y = torch.tensor(y, dtype=torch.int64).unsqueeze(0).cuda()

    config = AkatheV1Config()
    model = Transformer(config).cuda()

    optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.95), eps=10e-5, lr=3e-4)
    epochs = 1000

    for epoch in range(epochs):
        optimizer.zero_grad()
        _, loss = model(x, mask=generate_att_mask(x.size(1), x.device), y=y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

    torch.save(model.state_dict(), 'model.pth')