from app.web import app
from core.config import DEFAULT_HOST, DEFAULT_PORT


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT)
