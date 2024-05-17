import uvicorn
import torch
from transformers import pipeline, AutoTokenizer
from fastapi import FastAPI, Request

app = FastAPI()

@app.post("/")
async def create_item(request: Request):
    global pipe
    data = await request.json()
    prompt = data.get('prompt')
    print(prompt)

    messages = [
            {
                "role": "system",
                "content": "你是一个超级智者，名字叫shareAI-llama3，拥有优秀的问题解答能力。",
            },
            {"role": "user", "content": prompt}
    ]
    
    response = pipe(messages)
    # breakpoint()
    print(response)
    answer = {
        "response": response[-1]["content"],
        "status": 200,
    }
    return answer


if __name__ == '__main__':
    model_name_or_path = '/openbayes/home/baicai003/Llama3-Chinese-instruct-DPO-beta0___5'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    pipe = pipeline(
        "conversational", 
        model_name_or_path, 
        torch_dtype=torch.float16, 
        device_map="auto", 
        max_new_tokens=512, 
        do_sample=True,
        top_p=0.9, 
        temperature=0.6, 
        repetition_penalty=1.1,
        eos_token_id=tokenizer.encode('<|eot_id|>')[0]
    )

    uvicorn.run(app, host='0.0.0.0', port=9009)
    
