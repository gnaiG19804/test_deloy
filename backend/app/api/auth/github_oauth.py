from fastapi import APIRouter, Request, HTTPException, Depends
from starlette.responses import RedirectResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy import inspect
import httpx, secrets, os, base64, hashlib
from urllib.parse import urlencode
import datetime as dt
from app.core.config import settings
from app.core.security import create_access_token
from app.db.session import get_db
from app.models.user import User, AuthIdentity, Provider 
from starlette.responses import RedirectResponse

router = APIRouter(prefix="/auth/github", tags=["auth-github"])

GITHUB_AUTH_URL = "https://github.com/login/oauth/authorize"

GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"
GITHUB_EMAILS_URL = "https://api.github.com/user/emails"

USE_PKCE = True

def _pkce_pair():
    verifier = base64.urlsafe_b64encode(os.urandom(40)).decode().rstrip("=")
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    return verifier, challenge

def _get_oauth_map(request: Request) -> dict:
    return request.session.get("oauth_map", {})

def _set_oauth_map(request: Request, m: dict):
    request.session["oauth_map"] = m

@router.get("/login")
def github_login(request: Request):
    state = secrets.token_urlsafe(32)
    redirect_uri = settings.GITHUB_REDIRECT_URI

    oauth_map = _get_oauth_map(request)

    if USE_PKCE:
        verifier, challenge = _pkce_pair()
        oauth_map[state] = {"verifier": verifier}
    else:
        challenge = None
        oauth_map[state] = {}

    _set_oauth_map(request, oauth_map)

    params = {
        "client_id":settings.GITHUB_CLIENT_ID,
        "redirect_uri":redirect_uri,
        "state":state,
        "scope":"read:user user:email",
        "code_challenge":challenge,
        "code_challenge_method":"S256",
        "prompt":"consent"
    }
    if USE_PKCE:
        params.update({
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        })

    auth_url = f"{GITHUB_AUTH_URL}?{urlencode(params)}"

    print("GITHUB LOGIN state =", state)
    print("GITHUB >> redirect_uri =", redirect_uri)
    print("GITHUB >> auth_url =", auth_url)
    if USE_PKCE:
        print("GITHUB (PKCE) verifier =", oauth_map[state]["verifier"])

    return RedirectResponse(auth_url, status_code=302)

@router.get("/callback")
async def github_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    db: Session = Depends(get_db),
):
    if not code or not state:
        # raise HTTPException(400, "Thiếu code/state")
        url = f"{settings.FRONTEND_ORIGIN}/login?error=failed"
        return RedirectResponse(url, status_code=302)

    oauth_map: dict = request.session.get("oauth_map", {})
    entry = oauth_map.get(state)
    if entry is None:
        # raise HTTPException(400, "State không hợp lệ/expired")
        url = f"{settings.FRONTEND_ORIGIN}/login?error=failed"
        return RedirectResponse(url, status_code=302)

    verifier = entry.get("verifier") if USE_PKCE else None
    if USE_PKCE and not verifier:
        # raise HTTPException(400, "Thiếu code_verifier (session/PKCE)")
        # đăng nhập thất bại trả về trang login
        url = f"{settings.FRONTEND_ORIGIN}/login?error=failed"
        return RedirectResponse(url, status_code=302)

    token_payload = {
        "client_id": settings.GITHUB_CLIENT_ID,
        "client_secret": settings.GITHUB_CLIENT_SECRET,
        "code": code,
        "redirect_uri": settings.GITHUB_REDIRECT_URI,
    }
    if USE_PKCE:
        token_payload["code_verifier"] = verifier

    async with httpx.AsyncClient(timeout=15) as client:
        token_res = await client.post(
            GITHUB_TOKEN_URL,
            data=token_payload,
            headers={"Accept": "application/json"},
        )
        if token_res.status_code != 200:
            # raise HTTPException(400, f"Đổi token thất bại: {token_res.text}")
            url = f"{settings.FRONTEND_ORIGIN}/login?error=failed"
            return RedirectResponse(url, status_code=302)

        token_json = token_res.json()
        access_token = token_json.get("access_token")
        if not access_token:
            # raise HTTPException(400, f"Thiếu access_token từ GitHub: {token_json}")
            url = f"{settings.FRONTEND_ORIGIN}/login?error=failed"
            return RedirectResponse(url, status_code=302)

        ui_res = await client.get(
            GITHUB_USER_URL,
            headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
        )
        if ui_res.status_code != 200:
            raise HTTPException(400, f"Lấy user thất bại: {ui_res.text}")
        ui = ui_res.json()

        email = ui.get("email")
        if not email:
            em_res = await client.get(
                GITHUB_EMAILS_URL,
                headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
            )
            if em_res.status_code == 200:
                emails = em_res.json() or []
                email = next(
                    (e["email"] for e in emails if e.get("primary") and e.get("verified")),
                    (emails[0]["email"] if emails else None),
                )

    github_id = str(ui.get("id")) if ui.get("id") is not None else None
    login_name = ui.get("login")
    full_name = ui.get("name") or login_name
    avatar_url = ui.get("avatar_url")
    if not github_id:
        url = f"{settings.FRONTEND_ORIGIN}/login?error=failed"
        return RedirectResponse(url, status_code=302)

    provider_val = getattr(Provider.GITHUB, "value", Provider.GITHUB)

    try:
        identity = (
            db.query(AuthIdentity)
              .filter(AuthIdentity.provider == provider_val,
                      AuthIdentity.provider_user_id == github_id)
              .first()
        )

        if identity:
            db_user = identity.user
            identity.last_login_at = dt.datetime.now(dt.timezone.utc)
            db.commit()
        else:
            db_user = User(
                email=email,   
                full_name=full_name,  
                avatar_url=avatar_url,  
            )
            db.add(db_user)
            db.commit()
            db.refresh(db_user)

            identity = AuthIdentity(
                user_id=db_user.id,
                provider=provider_val,
                provider_user_id=github_id,
                last_login_at=dt.datetime.now(dt.timezone.utc),
            )
            db.add(identity)
            db.commit()

    except IntegrityError as e:
        db.rollback()
        identity = (
            db.query(AuthIdentity)
              .filter(AuthIdentity.provider == provider_val,
                      AuthIdentity.provider_user_id == github_id)
              .first()
        )
        if identity:
            db_user = identity.user
        else:
            raise HTTPException(500, f"Lỗi lưu DB: {e}")
    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"Lỗi lưu DB: {e}")

    jwt_token = create_access_token(str(db_user.id))

    resp = RedirectResponse(settings.FRONTEND_ORIGIN, status_code=302)

    resp.set_cookie(
        key="access_token",
        value=jwt_token,
        httponly=True,
        secure=False, 
        samesite="lax",
        path="/",
        max_age=3600,
    )
    return resp