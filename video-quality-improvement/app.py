from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def scrape():
    return {"status": "Hello from Scrapper"}