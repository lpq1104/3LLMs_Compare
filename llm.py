from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM

# 这里替换成你当前测试的模型路径
model_name = "/mnt/data/Qwen-7B-Chat"
# model_name = "/mnt/data/chatglm3-6b"
# model_name = "/mnt/data/Baichuan2-7B-Chat"

# 用作业里的歧义题
#prompt = "请说出以下两句话区别在哪里？1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少"
#prompt = "请说出以下两句话区别在哪里？单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上"
#prompt = "他知道我知道你知道他不知道吗？ 这句话里，到底谁不知道"
#prompt = "明明明明明白白白喜欢他，可她就是不说。 这句话里，明明和白白谁喜欢谁？"
prompt = "领导：你这是什么意思？ 小明：没什么意思。意思意思。 领导：你这就不够意思了。 小明：小意思，小意思。领导：你这人真有意思。 小明：其实也没有别的意思。 领导：那我就不好意思了。 小明：是我不好意思。请问：以上“意思”分别是什么意思。"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).eval()

inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
