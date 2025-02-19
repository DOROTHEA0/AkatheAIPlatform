from llm.config import AkatheV1Config
from llm.model import *
from llm.utils import count_parameters, generate_att_mask
import torch
from torch import optim
from transformers import PreTrainedTokenizerFast

def test_llm():
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="checkpoints/tokenizer/tokenizer.json")
    text = '''人工智能正在快速发展。近年来，大型语言模型在自然语言处理领域取得了突破性进展。
            这些模型不仅能够理解和生成文本，还能进行编程、数学计算和创意写作。
            不过，我们也要警惕AI发展带来的潜在风险和伦理问题。
            让我们共同努力，确保AI技术造福人类社会！            
            2024年，全球AI市场规模预计将达到5000亿美元。
            深度学习模型的参数量从10年前的百万级别发展到了现在的千亿级别。
            ChatGPT的出现让更多人认识到了AI的潜力。'''
    encoded = tokenizer(text)
    l = len(encoded["input_ids"])
    x = encoded["input_ids"][0: l - 1]
    y = encoded["input_ids"][1: l]

    x = torch.tensor(x, dtype=torch.int64).unsqueeze(0).cuda()
    y = torch.tensor(y, dtype=torch.int64).unsqueeze(0).cuda()

    config = AkatheV1Config()
    model = Transformer(config).cuda()

    optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.95), eps=10e-5, lr=3e-4)
    epochs = 500

    for epoch in range(epochs):
        optimizer.zero_grad()
        _, loss = model(x, mask=generate_att_mask(x.size(1), x.device), y=y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')
    torch.save(model.state_dict(), 'model.pth')

def generate_text(model, tokenizer, text_in="人工智能", max_len=100):
    encoded = tokenizer(text_in)
    l = len(encoded["input_ids"])
    x = encoded["input_ids"]
    x = torch.tensor(x, dtype=torch.int64).unsqueeze(0).cuda()
    model.eval()
    with torch.no_grad():
        for i in range(max_len):
            logits, _ = model(x, use_cache=True, y=None)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            x_next = torch.argmax(probs, dim=-1)
            print(tokenizer.decode(x_next.item()), end="")
            x = torch.cat((x, x_next.unsqueeze(0)), dim=1)

if __name__ == '__main__':
    #test_llm()
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="checkpoints/tokenizer/tokenizer.json")
    model = Transformer(AkatheV1Config()).cuda()
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    generate_text(model, tokenizer)

