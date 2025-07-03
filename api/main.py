from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
import httpx

app = FastAPI()

SERVICE_MAP = {
    "auth": "http://auth:5001",
    "scrapper": "http://scrapper:5002",
    "ai-generation": "http://ai-generation:5003",
    # Extend with other services...
}

@app.get("/")
async def root():
    return {"message": "Hello from API Gateway"}

@app.api_route("/{service}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy(service: str, path: str, request: Request):
    if service not in SERVICE_MAP:
        return JSONResponse(status_code=404, content={"error": "Service not found"})

    # Construct full target URL
    target_url = f"{SERVICE_MAP[service]}/{path}"

    # Prepare headers and body from the incoming request
    headers = {key: value for key, value in request.headers.items() if key.lower() != 'host'}
    body = await request.body()

    # Forward the request to the respective microservice
    async with httpx.AsyncClient() as client:
        try:
            response = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
                params=request.query_params
            )
        except httpx.RequestError as exc:
            return JSONResponse(status_code=502, content={"error": f"Service unavailable: {exc}"})

    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.headers.get("content-type")
    )
