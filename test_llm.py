from llm.config import AkatheV1Config
from llm.model import *
from llm.utils import count_parameters, generate_att_mask
import torch
from torch import optim
from transformers import PreTrainedTokenizerFast




if __name__ == '__main__':
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="checkpoints/tokenizer/tokenizer.json")
    text = '''想要做村民的话，必须等到十级才能成为村长。想转职的话，又必须达到十级。这特么玩我呢吧？“除了刚才的方法，还有快速升级的手段吗？”“啊…也不是没有就是了……”“拜托了，请告诉我！”我把双手合十举过头顶。“好吧。”少女似乎又开始为难起来，“方法就是‘心意共享’。”“……抱歉，我失忆了。”“呃…如果是两位冒险者之间，只需要达成短期的‘互不背叛’契约，就可以在契约生效期间共享心意。”哦~听上去像是组队啊！“那就拜托了，请和我契约！”我大脑一热，几乎不经思考便再次向她俯首恳求，“啊……这个……”看到她这扭捏的模样，我脑子一个激灵反应了过来：组队的话，经验会被分配。而我这个完全没有战斗力只能划水的咸鱼，只会拖慢她的升级进度。大号带小号刷本还要给好处呢，可我身上却什么都没有。何况人家已经救了我一命了，又怎么好这样麻烦她？“这个…真对不起，我知道这个要求太过份了，当我没说过好了。”我语气尽可能的轻松。只要到达城镇，总有办法找到人组队的。“不、不是这样！请不要误会，不是我个人的原因！”她听到我说的话，突然有些慌张起来，“我的意思是……我刚才说了吧，必须两个冒险者之间，才能通过短期‘互不背叛’契约共享心意。可你是村民，属于这个世界的原住民。原住民再如何转职，也是无法成为冒险者的。”“啥？”我瞪大着眼睛看着她。本以为终于找到了拜托咸鱼命运的办法，结果从这个世界的设定上我就已经翻不了身了是吗？“你如果真的很想到处冒险的话，可以转职为‘佣兵’，让冒险者雇用你，之后就可以跟着冒险者四处探险并且获得心意了。”“……”我突然脊背有些发寒。不是因为这个条件太难，而是我想起了我曾经做过的事情。在我以前玩过的RPG游戏中，我经常会雇佣一些价格低廉的低级佣兵，然后每逢要进入危险地图的时候，都先把这些冤大头派进去探路。这还只是其中一种，我玩游戏的时候，可没少残害佣兵们。不知道多少的佣兵被我浪死在了种种危险的地方。可是现在，我却面临着成为佣兵的选择。这难道就是“一报还一报”？啊，那些程序中游荡的冤魂啊，放过我吧！“成为佣兵的话，会完全失去自由吗？”“只要不是有关尊严底线以及明摆着的死亡之类的命令，几乎都会被契约强制执行。不过也有那种把整个人都卖出去的女佣兵，以及豁出性命多拿些钱的亡命徒。”“这种契约是永久的吗？”“主从契约可选择时限，最低时限是……一个月。”我顿时陷入了纠结。我并不是一个底线很高的人，扫扫厕所擦擦马桶绝对不是什么难事。但就算我再能忍，这个“明摆着的死亡”也有些过于暧昧了。虽然不能命令佣兵直接自杀，但间接派去送死之类的还不是小事？我可不想刚踏上冒险的道路就被糊里糊涂的命令害了性命。“啊，真是难办啊……”我苦恼地挠着头，一时间不知怎么办才好。林间有些寂静，除了我们两人踩过落叶发出的沙沙声之外，只有偶尔从树梢上传来的一声鸟鸣。我皱着眉头冥思苦想，少女则在一旁默不作声。“我决定了！”我像是给自己鼓气似的大喊一声，把一旁的少女吓了一跳。“如果你不嫌弃的话，请和我契约吧！”“嫌弃？不、不嫌弃！”她被我说得一愣，“应该说…我作为冒险者，还不够成熟，我…没问题的吗？”“绝对没问题！”靠，我都快把这个世界的系统摸透了，你再蠢，我都能把你变成RPG老手！“那、那就开始吧。”她停了下来，站在了我身后，轻轻把手掌搭在我背后那块有痕刻的位置上。一股热量传来，我感到我的背部正在凝聚着力量。“汝可愿意？”金发的少女语气严肃地说了这么一句不知头尾的话。“愿意。”话一出口，我突然就开始后悔了。万一这个少女只是看上去单纯善良，切开来里面却是黑的怎么办？我会度过生不如死的一个月的吧？但貌似已经来不及了。冥冥之中，我能感知到一条若有若无的细线，将我和身后的少女隐隐约约地连接起来。“好了，契约以达成。”她松了口气，将手从我的背上放下。不知道她这是不是第一次契约，所以才看上去有些紧张。“这么简单……还真是方便啊。”我摸着背后的村民痕刻，嘟囔了一句。“我们之间的意向已经达成，契约起来自然就简单多了。”“我还以为要念一些‘愿此身终为汝剑’之类撑场面的话呢。”“要是在佣兵大厅里招募佣兵的话，的确是要说一些场面话。不过我们现在就免了吧。”她见我不再接话茬，侧过头小心翼翼地看了我一眼。“你……是在不安吗？”“哇啊！”我正在思考着这个金发少女黑化的可能性，突然被一语道破心机，不由得慌乱起来。“你、你会读心吗？”“什么嘛！我又不是暗影牧师！只是……作为主从签约的主方，我能够大致的感知到你的心情如何，是高兴还是沮丧，这样才比较方便相处嘛……”“你说的那是主人与仆人相处的方式好吗？啊，真是的！早知道这样就不做这个契约了！”我发着牢骚。不过就算真的是读心，其实我估计也是会接受这个契约的。不能转职，我和一盘咸鱼有什么区别？少了个盘子吗？“真、真对不起，都怪我没有说清楚。”少女羞赧地垂下了头。“啊……我这人比较爱发牢骚，别介意。”我突然觉得，粉切黑的可能性不大。“哦，对了，现在我也属于战斗单位了吧！”我想到了什么，突然兴奋起来，“请给我装备吧！”“装备……”少女抿了抿嘴，神色又开始为难起来。ps.这几天家里有事，作者君申请记账'''
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