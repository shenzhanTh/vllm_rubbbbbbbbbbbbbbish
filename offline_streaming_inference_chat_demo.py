from vllm.sampling_params import SamplingParams
from vllm.engine.async_llm_engine import AsyncEngineArgs, AsyncLLMEngine
import asyncio
from vllm.utils import FlexibleArgumentParser
from transformers import AutoTokenizer, AutoModel
import logging
vllm_logger = logging.getLogger("vllm")
vllm_logger.setLevel(logging.WARNING)


parser = FlexibleArgumentParser()
parser = AsyncEngineArgs.add_cli_args(parser)
args = parser.parse_args()

# chat = [
#   {"role": "user", "content": "Hello, how are you?"},
#   {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
#   {"role": "user", "content": "I'd like to show off how chat templating works!"},
# ]

# tokenizer =  AutoTokenizer.from_pretrained("/models/llama2/Llama-2-7b-chat-hf")
# aaaa = tokenizer.chat_template
# print(aaaa)
engine_args = AsyncEngineArgs.from_cli_args(args)
engine = AsyncLLMEngine.from_engine_args(engine_args)


model_name = args.model.split("/")[-1] if args.model.split("/")[-1] !=""  else args.model.split("/")[-2]
print(f"欢迎使用{model_name}模型,输入内容即可进行对话,stop 终止程序")


def build_prompt(history):
    prompt = ""
    for query, response in history:
        prompt += f"\n\n用户:{query}"
        prompt += f"\n\n{model_name}:{response}"
    return prompt


history = "<s>[INST] Hello, how are you? [/INST] I'm doing great. How can I help you today?</s>" 

while True:
     query = input("\n用户:")
     if query.strip() == "stop":
          break 
     query = history + "<s>[INST] " + query + " [/INST]"

     
     example_input = {
     "prompt": query,
     "stream": False, 
     "temperature": 0.0,
     "request_id": 0,
     }

     results_generator = engine.generate(
     example_input["prompt"],
     SamplingParams(temperature=example_input["temperature"], max_tokens=100),
     example_input["request_id"]
     )

     start = 0
     end = 0
     last = ""
     async def process_results():
          async for  output in results_generator: 
               global end 
               global start 
               global last
               print(output.outputs[0].text[start:], end="", flush=True)
               length = len(output.outputs[0].text)
               start = length
               last = output.outputs[0].text
     
     asyncio.run(process_results())
     history += "<s>[INST] " + query + " [/INST]" + last + "</s>"
print()
#print(history)

