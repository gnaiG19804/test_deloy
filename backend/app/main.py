from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.db.session import SessionLocal
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.auth.google_oauth import router as google_api
from app.api.auth.github_oauth import router as github_api
from app.api.user.user_api import router as user_api
from app.api.user.user_api import router as logout_api
from app.api.user.summary_api import router as summary_api

app = FastAPI(title="Text Summarizer Backend", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    SessionMiddleware,
    secret_key=settings.SECRET_KEY, 
    session_cookie="sid",           
    same_site="lax",              
    https_only=False,              
    max_age=60 * 60 * 2,            
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/test-db")
def db_test(db: Session = Depends(get_db)):
    try:
        result = db.execute(text("SELECT 1")).scalar()
        return {"status": "ok", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}
@app.on_event("startup")
def _warmup():
    from app.llm.summarizer import load_model
    try:
        load_model()
        print("[startup] model ready (cached)")
    except Exception as e:
        print(f"[startup] warmup failed: {e}")

app.include_router(google_api)
app.include_router(github_api)
app.include_router(user_api)
app.include_router(logout_api)
app.include_router(summary_api)
