# isort: skip_file
import copy
import warnings
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional

import streamlit as st
import torch

from transformers.utils import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig  # isort: skip
from peft import PeftModel
from threading import Thread
from transformers import TextIteratorStreamer

logger = logging.get_logger(__name__)
st.set_page_config(page_title="Llama3-Chinese")

import argparse

@dataclass
class GenerationConfig:
    # this config is used for chat to provide more diversity
    max_length: int = 8192
    max_new_tokens: int = 600
    top_p: float = 0.8
    temperature: float = 0.8
    do_sample: bool = True
    repetition_penalty: float = 1.05

def on_btn_click():
    del st.session_state.messages

@st.cache_resource
def load_model(model_name_or_path, adapter_name_or_path=None, load_in_4bit=False):
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    else:
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map='auto',
        quantization_config=quantization_config
    )
    if adapter_name_or_path is not None:
        model = PeftModel.from_pretrained(model, adapter_name_or_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    return model, tokenizer

def prepare_generation_config():
    with st.sidebar:
        st.title('超参数面板')
        # 大输入框
        system_prompt_content = st.text_area('系统提示词',
            "你是一个调皮活泼的中文智者，名字叫shareAI-Llama3，喜欢用有趣的语言和适当的表情回答问题。",
            height=200,
            key='system_prompt_content'
        )
        max_new_tokens = st.slider('最大回复长度', 100, 8192, 1020, step=8)
        top_p = st.slider('Top P', 0.0, 1.0, 0.8, step=0.01)
        temperature = st.slider('温度系数', 0.0, 1.0, 0.6, step=0.01)
        repetition_penalty = st.slider("重复惩罚系数", 1.0, 2.0, 1.07, step=0.01)
        st.button('重置聊天', on_click=on_btn_click)

    generation_config = GenerationConfig(max_new_tokens=max_new_tokens,
                                         top_p=top_p,
                                         temperature=temperature,
                                         repetition_penalty=repetition_penalty,
                                        )

    return generation_config

system_prompt = '<|begin_of_text|><<SYS>>\n{content}\n<</SYS>>\n\n'
user_prompt = '<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>'
robot_prompt = '<|start_header_id|>assistant<|end_header_id|>\n\n{robot}<|eot_id|>'
cur_query_prompt = '<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'

def combine_history(prompt):
    messages = st.session_state.messages
    total_prompt = ''
    for message in messages:
        cur_content = message['content']
        if message['role'] == 'user':
            cur_prompt = user_prompt.format(user=cur_content)
        elif message['role'] == 'robot':
            cur_prompt = robot_prompt.format(robot=cur_content)
        else:
            raise RuntimeError
        total_prompt += cur_prompt

    system_prompt_content = st.session_state.system_prompt_content
    system = system_prompt.format(content=system_prompt_content)
    total_prompt = system + total_prompt + cur_query_prompt.format(user=prompt)
    
    return total_prompt

def main(model_name_or_path, adapter_name_or_path):
    # torch.cuda.empty_cache()
    print('load model...')
    model, tokenizer = load_model(model_name_or_path, adapter_name_or_path=adapter_name_or_path, load_in_4bit=False)
    print('load model end.')

    st.title('Llama3-Chinese')

    generation_config = prepare_generation_config()

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Accept user input
    if prompt := st.chat_input('解释一下Vue的原理'):
        # Display user message in chat message container
        with st.chat_message('user'):
            st.markdown(prompt)
        real_prompt = combine_history(prompt)
        # Add user message to chat history
        st.session_state.messages.append({
            'role': 'user',
            'content': prompt,
        })

        with st.chat_message('robot'):
            message_placeholder = st.empty()
            
            inputs = tokenizer([real_prompt], return_tensors='pt').to(model.device)
            streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
            
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=generation_config.max_new_tokens,
                do_sample=generation_config.do_sample,
                top_p=generation_config.top_p,
                temperature=generation_config.temperature,
                repetition_penalty=generation_config.repetition_penalty,
            )

            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            response = ''
            for token in streamer:
                response += token
                message_placeholder.markdown(response + '▌')
            message_placeholder.markdown(response)

        # Add robot response to chat history
        st.session_state.messages.append({
            'role': 'robot',
            'content': response,
        })
        torch.cuda.empty_cache()

if __name__ == '__main__':

    import sys
    model_name_or_path = sys.argv[1]
    if len(sys.argv) >= 3:
        adapter_name_or_path = sys.argv[2]
    else:
        adapter_name_or_path = None
    main(model_name_or_path, adapter_name_or_path)
