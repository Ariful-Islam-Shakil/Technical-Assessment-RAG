from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

def generate_answer(prompt: str , temperature: float = 0.3) -> str:
    client = Groq(api_key=GROQ_API_KEY)
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Response the queries based on the given set of contexts.Also Cite sources by [doc_title] in each paragrapgh if required"},
        {"role": "user", "content": prompt}
    ]
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=8192
    )
    return completion.choices[0].message.content

