from app.web import app
from core.config import DEFAULT_HOST, DEFAULT_PORT


if __name__ == "__main__":
    import uvicorn

    # proxy_headers + forwarded_allow_ips let the app see the real https scheme
    # when running behind a reverse proxy (e.g. Hugging Face Spaces), so generated
    # URLs use https and the browser doesn't block assets as mixed content.
    uvicorn.run(
        app,
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
