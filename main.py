from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv
import traceback

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env file")

# -------------------------------
# Configure Gemini
# -------------------------------
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(
    title="Intentra API",
    description="Intentra – Extracts real intent from words and upgrades AI prompts",
    version="1.0"
)

# -------------------------------
# Enable CORS
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Request model (THIS ENABLES SWAGGER INPUT)
# -------------------------------
class IntentRequest(BaseModel):
    input: str

# -------------------------------
# Main API endpoint
# -------------------------------
@app.post("/api/intentra")
async def intentra_api(req: IntentRequest):
    try:
        user_input = req.input.strip()

        if not user_input:
            raise HTTPException(status_code=400, detail="Input cannot be empty")

        # 1️⃣ Extract intent
        intent_prompt = (
            "Extract the real intent from the user's input in ONE clear sentence.\n\n"
            f"Input: {user_input}"
        )
        intent = model.generate_content(intent_prompt).text.strip()

        # 2️⃣ Optimize prompt
        optimize_prompt = (
            "Convert the following intent into a high-quality, structured AI prompt:\n\n"
            f"{intent}"
        )
        optimized_prompt = model.generate_content(optimize_prompt).text.strip()

        # 3️⃣ Generate final answer
        final_prompt = (
            "Answer the following prompt clearly, simply, and accurately:\n\n"
            f"{optimized_prompt}"
        )
        answer = model.generate_content(final_prompt).text.strip()

        return {
            "intent": intent,
            "prompt": optimized_prompt,
            "answer": answer
        }

    except Exception as e:
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }

# -------------------------------
# Health check
# -------------------------------
@app.get("/")
def root():
    return {"message": "Intentra Gemini API is running"}
