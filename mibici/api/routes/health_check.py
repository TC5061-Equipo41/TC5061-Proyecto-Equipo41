from fastapi import APIRouter, Request

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health(request: Request):
    return {
        "status": "ok",
        "model_loaded": getattr(request.app.state, "model", None) is not None,
    }

