from fastapi import HTTPException, Request


def get_model(request: Request):
    """
    Dependency to get the model from the app state.
    """
    if request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return request.app.state.model


def get_app_settings(request: Request):
    """
    Dependency to get the application settings from the app state.
    """
    return request.app.state.settings


def get_device(request: Request):
    """
    Dependency to get the compute device from the app state.
    """
    return request.app.state.device
