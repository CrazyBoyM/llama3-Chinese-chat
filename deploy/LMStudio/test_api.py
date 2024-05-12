from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

completion = client.chat.completions.create(
  model="shareAI/llama3-dpo-zh",
  messages=[
    {"role": "system", "content": "you are a helpful bot."},
    {"role": "user", "content": "讲个笑话"}
  ],
  temperature=0.7,
  stop=["<|eot_id|>"],
)

print(completion.choices[0].message)
