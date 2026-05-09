# 大语言模型部署实践报告

吕佩琦 2453612

---
## 大语言模型概述
 **大语言模型（Large Language Model，LLM）** 是一类基于深度学习的人工智能模型，主要用于理解和生成自然语言。它通过在海量文本数据上进行训练，学习语言中的语法、语义、知识以及上下文关系，从而能够完成问答、翻译、写作、代码生成、文本总结等多种任务。
目前主流的大语言模型大多建立在 **Transformer** 架构之上。Transformer 的核心是“注意力机制（Attention）”，它能够让模型在处理一句话时，动态关注与当前词最相关的其他词，从而更好地理解上下文。

**大语言模型的基本训练过程：**
- **预训练（Pretraining）**
在互联网文本、书籍、论文等海量数据上训练，目标通常是“预测下一个词”，模型在这个阶段学习语言规律和世界知识。
- **微调与对齐（Fine-tuning / Alignment）**
使用特定任务数据进一步训练，通过人工反馈（RLHF）让回答更符合人类习惯，提升安全性、准确性和可用性。

**模型规模：**
大语言模型的参数数量非常庞大，从数亿到数千亿不等。参数数量越多，模型的表现通常越好，但也需要更多的计算资源和更长的训练时间。

**应用领域：**
- 文本生成：如写作辅助、新闻生成、内容创作等。
- 语言翻译：将一种语言的文本翻译成另一种语言。
- 问答系统：回答用户的问题，如搜索引擎、客服系统等。
- 对话机器人：与用户进行自然语言对话。
- 文本分析：情感分析、主题分类等。

**局限性：**
- 幻觉：会生成看似合理但错误的信息。
- 推理不稳定：复杂逻辑问题可能出错。
- 知识时效性有限：训练后无法自动获得新知识。
- 计算资源消耗巨大：训练和推理成本高。
- 安全与偏见问题：可能继承训练数据中的偏差。
---

## 利用魔搭社区部署大语言模型
1. 注册并登录魔搭平台 ModelScope（https://www.modelscope.cn/home）。
2. 关联阿里云账号，获得免费的CPU云计算资源。
3. 启动 Notebook 准备。
![alt text](image-26.png)
4. 打开Terminal终端命令行环境
![alt text](image-27.png)
## 环境搭建
- **安装conda：**
```
cd /opt
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda
echo 'export PATH="/opt/conda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
conda --version
```
![alt text](image.png)

- **接受 Anaconda 官方源的服务条款后，创建并激活环境：**
```
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```
```
conda create -n llm_env python=3.10 -y
source /opt/conda/etc/profile.d/conda.sh
conda activate llm_env
```
- **安装依赖：**
```
pip install -U pip setuptools wheel

pip install torch==2.3.0+cpu torchvision==0.18.0+cpu --index-url https://download.pytorch.org/whl/cpu

pip install "intel-extension-for-transformers==1.4.2" "neural-compressor==2.5" "transformers==4.33.3" modelscope==1.9.5 pydantic==1.10.13 sentencepiece tiktoken einops transformers_stream_generator uvicorn fastapi yacs setuptools_scm
```
可选：安装 tqdm、huggingface-hub 等增强体验
```
pip install fschat --use-pep517
pip install tqdm huggingface-hub
```

---

## 大模型实践
### 下载大模型到本地
- **切换到数据目录：**
```
cd /mnt/data
```
- **下载对应大模型：**
一次最好只跑下载一个大模型，否则会存储不足。
```
git clone https://www.modelscope.cn/qwen/Qwen-7B-Chat.git
git clone https://www.modelscope.cn/ZhipuAI/chatglm3-6b.git
git clone https://www.modelscope.cn/baichuan-inc/Baichuan2-7B-Chat.git
```
![alt text](image-11.png)
![alt text](image-4.png)
![alt text](image-12.png)

### 构建实例
- **切换工作目录：**
```
cd /mnt/workspace
```
- **编写推理脚本`llm.py`：**
```
from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM

# 可替换成当前测试的模型路径
model_name = "/mnt/data/chatglm3-6b"

# 可替换成当前问题
prompt = "请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).eval()

inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
```
- **运行实例：**
```
python llm.py
```
### 问答测试
在相应环境的jupyter notebook中测试不同大语言模型的特点，并在每个大语言模型中问出以下问题，查看不同模型的回答：
1. 请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少
2. 请说出以下两句话区别在哪里？单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上
3. 他知道我知道你知道他不知道吗？ 这句话里，到底谁不知道
4. 明明明明明白白白喜欢他，可她就是不说。 这句话里，明明和白白谁喜欢谁？
5. 领导：你这是什么意思？ 小明：没什么意思。意思意思。 领导：你这就不够意思了。 小明：小意思，小意思。领导：你这人真有意思。 小明：其实也没有别的意思。 领导：那我就不好意思了。 小明：是我不好意思。请问：以上“意思”分别是什么意思。
### 测试结果
#### Qwen-7B-Chat：
![alt text](image-14.png)
![alt text](image-16.png)
![alt text](image-17.png)
![alt text](image-25.png)
![alt text](image-19.png)

#### chatglm3-6b：
![alt text](image-6.png)
![alt text](image-7.png)
![alt text](image-8.png)
![alt text](image-9.png)
![alt text](image-10.png)

#### Baichuan2-7B-Chat：
![alt text](image-20.png)
![alt text](image-21.png)
![alt text](image-22.png)
![alt text](image-23.png)
![alt text](image-24.png)

---

## 大语言模型横向比对分析
根据示例问题和大语言模型的输出，我们对这三个大语言模型进行横向比对：
- **语言流畅性和可读性**
  - Qwen-7B-Chat：语言表达较为直接，有时语气更加口语化。在处理复杂语义或含有双关语的问题时，倾向于提供更加简洁和直观的回答。
  - chatglm3-6b：输出较为正式和条理清晰。它通常在回答中提供详细的解释和论述，尤其在需要详细解释的问题上表现良好。
  - Baichuan2-7B-Chat：输出较为直接且简洁，有时较为口语化。
- **处理复杂问题的能力**
  - Qwen-7B-Chat：在解释复杂概念或回答技术性问题时，有时倾向于过度简化。
  - chatglm3-6b：在处理需要多层次理解和逻辑推理的问题时，提供了更深入的分析和解释。
  - Baichuan2-7B-Chat：在简单语境歧义问题上表现尚可，但对于复杂问题的判断能力一般，有时无法正确判断，缺乏处理这类语言分析任务的专业水平。
- **准确性和专业知识**
  - Qwen-7B-Chat：在处理专业问题时，能够准确快速地给出核心信息，尽管有时细节不如chatglm3-6b丰富。
  - chatglm3-6b：在需要专业知识的回答中，提供了详尽的解释，显示出较好的知识储备。
  - Baichuan2-7B-Chat：仅能正确理解少量表层语境歧义，对多数典型中文歧义问题存在错误。
- **回答的一致性和逻辑性**
  - Qwen-7B-Chat：回答简洁且直接，逻辑通常清晰，但有时可能在更复杂的逻辑链条上表现略逊一筹。
  - chatglm3-6b：在逻辑推理和答案的一致性方面表现较好，尤其是在需要连续逻辑步骤的问题上。
  - Baichuan2-7B-Chat：在简单语境歧义问题上保持了基础的逻辑框架，对复杂歧义句的处理逻辑前后不一致

总体而言，智谱 chatglm3-6b 适合处理需要深度知识和详细解释的场合；千问 Qwen-7B-Chat 更适合日常对话和需要快速答案的情境；百川 Baichuan2-7B-Chat 对计算资源要求较高，且性能不如前两个。