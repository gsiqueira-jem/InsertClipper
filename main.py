from fastapi import FastAPI
from routes import router  # your router from routes.py

app = FastAPI()
app.include_router(router)

# Optional: Run uvicorn programmatically (but usually done via CLI)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)