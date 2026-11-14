from fastapi import APIRouter, Request

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health(request: Request):
    models = getattr(request.app.state, "models", {})
    return {
        "status": "ok",
        "models_loaded": {key: model.get("loaded") is not None for key, model in models.items()},
    }
