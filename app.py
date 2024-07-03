from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Any, Dict
from llama_cpp import Llama
import pandas as pd
from pydantic import BaseModel
from langchain.llms.base import LLM
import time
import uvicorn

#import uvicorn-

# Constants
MODEL_NAME = 'mellogpt.Q3_K_S.gguf'
MODEL_PATH = 'D:\\python\\final_mental\\chatbot\\mellogpt.Q3_K_S.gguf'
KNOWLEDGE_BASE_FILE = "mentalhealth.csv"
NUM_THREADS = 8

# Initialize FastAPI app
app = FastAPI()

# Load Knowledge Base
def load_knowledge_base():
    df = pd.read_csv(KNOWLEDGE_BASE_FILE)
    return dict(zip(df['Questions'].str.lower(), df['Answers']))

knowledge_base = load_knowledge_base()

# Query data model
class Query(BaseModel):
    query: str

# Custom LLM Class
class CustomLLM(LLM):
    model_name = MODEL_NAME

    def _call(self, prompt: str) -> str:
        p = f"Human: {prompt} Assistant: "
        llm = Llama(model_path=MODEL_PATH, n_threads=NUM_THREADS)
        try:
            output = llm(p, max_tokens=512, stop=["Human:"], echo=True)['choices'][0]['text']
            response = output[len(p):]
            return response
        except Exception as e:
            raise RuntimeError("Failed to process the LLM request.")

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"

llm = CustomLLM()

# Add CORS middleware
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
    "https://localhost:5173",
    # Add other origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# API Endpoint for querying the chatbot
@app.post('/query')
async def handle_query(query: Query):
    user_input = query.query.lower()
    print("Request Recieved" , user_input)
    answer = knowledge_base.get(user_input)
    # time.sleep(5)
    # answer = "how can i help you";
    print("Calculating Answer")

    if answer:
        response = {"role": "assistant", "content": answer}
        print("Answer Recieved", answer)
    else:
        try:
            print("Calling Model Again")
            response_text = llm._call(prompt=user_input)
            response = {"role": "assistant", "content": response_text}
        except Exception as e:
            print("Error:", str(e))
            raise HTTPException(status_code=500, detail="Internal Server Error: Error processing your request")

    return JSONResponse(content=response)

# Running the FastAPI app with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)