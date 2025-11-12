from fastapi import APIRouter, Request, HTTPException, Depends
from starlette.responses import RedirectResponse
from sqlalchemy.orm import Session
from sqlalchemy import inspect
import httpx, secrets, os, base64, hashlib
from urllib.parse import urlencode
import datetime as dt
from app.core.config import settings
from app.core.security import create_access_token
from app.db.session import get_db
from app.models.user import User, AuthIdentity, Provider 

router = APIRouter(prefix="/auth/google", tags=["auth-google"])

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
SCOPES = ["openid", "email", "profile"]

USE_PKCE = True

def _pkce_pair():
    verifier = base64.urlsafe_b64encode(os.urandom(40)).decode().rstrip("=")
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    return verifier, challenge

@router.get("/login")
def google_login(request: Request):
    state = secrets.token_urlsafe(32)
    redirect_uri = settings.GOOGLE_REDIRECT_URI

    if USE_PKCE:
        verifier, challenge = _pkce_pair()
        request.session["oauth"] = {"state": state, "verifier": verifier}
    else:
        request.session["oauth"] = {"state": state}

    request.session['oauth']={
        "state": state,
        "verifier": verifier,
    }

    params = {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "state": state,
        "prompt": "consent",
        "access_type": "offline",
    }
    if USE_PKCE:
        params.update({
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        })

    auth_url = f"{GOOGLE_AUTH_URL}?{urlencode(params)}"

    # Debug
    print("LOGIN state =", state)
    print("LOGIN >> redirect_uri =", redirect_uri)
    print("LOGIN >> google_auth_url =", auth_url)
    if USE_PKCE:
        print("LOGIN (PKCE) verifier =", request.session["oauth"]["verifier"])

    return RedirectResponse(auth_url, status_code=302)

@router.get("/callback")
async def google_callback(request: Request, db: Session = Depends(get_db)):
    code = request.query_params.get("code")
    state = request.query_params.get("state")

    # Debug
    print("HOST:", request.headers.get("host"))
    print("REDIRECT_URI (settings):", settings.GOOGLE_REDIRECT_URI)
    print("COOKIE SESSION PRESENT:", "session" in request.cookies)
    print("CALLBACK code present:", bool(code), "| state:", state)

    if not code or not state:
        # raise HTTPException(400, "Thiếu code hoặc state")
        url = f"{settings.FRONTEND_ORIGIN}/login?error=failed"
        return RedirectResponse(url, status_code=302)

    sess = request.session.get("oauth")
    print("SESSION OAUTH:", sess)
    if not sess or sess.get("state") != state:
        # raise HTTPException(400, "State không hợp lệ (không khớp session)")
        url = f"{settings.FRONTEND_ORIGIN}/login?error=failed"
        return RedirectResponse(url, status_code=302)

    verifier = sess.get("verifier") if USE_PKCE else None
    if USE_PKCE and not verifier:
        # raise HTTPException(400, "Thiếu code_verifier (session/PKCE)")
        url = f"{settings.FRONTEND_ORIGIN}/login?error=failed"
        return RedirectResponse(url, status_code=302)

    #Lấy thông tin user
    token_payload = {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "client_secret": settings.GOOGLE_CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
    }
    if USE_PKCE:
        token_payload["code_verifier"] = verifier

    async with httpx.AsyncClient(timeout=15) as client:
        token_res = await client.post(
            GOOGLE_TOKEN_URL,
            data=token_payload,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
        )
        print("TOKEN RES CODE:", token_res.status_code)
        print("TOKEN RES BODY:", token_res.text)

        if token_res.status_code != 200:
            # raise HTTPException(400, f"Đổi token thất bại: {token_res.text}")
            url = f"{settings.FRONTEND_ORIGIN}/login?error=failed"
            return RedirectResponse(url, status_code=302)

        token_json = token_res.json()
        access_token = token_json.get("access_token")
        if not access_token:
            # raise HTTPException(400, f"Thiếu access_token từ Google: {token_json}")
            url = f"{settings.FRONTEND_ORIGIN}/login?error=failed"
            return RedirectResponse(url, status_code=302)

        ui_res = await client.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        print("USERINFO RES CODE:", ui_res.status_code)
        print("USERINFO RES BODY:", ui_res.text)
        if ui_res.status_code != 200:
            # raise HTTPException(400, f"Lấy userinfo thất bại: {ui_res.text}")
            url = f"{settings.FRONTEND_ORIGIN}/login?error=failed"
            return RedirectResponse(url, status_code=302)
        ui = ui_res.json()

    google_sub = ui.get("sub")
    email = ui.get("email")
    full_name = ui.get("name")
    avatar = ui.get("picture")
    if not google_sub:
        url = f"{settings.FRONTEND_ORIGIN}/login?error=failed"
        return RedirectResponse(url, status_code=302)
    if not email:
        url = f"{settings.FRONTEND_ORIGIN}/login?error=failed"
        return RedirectResponse(url, status_code=302)

    try:
        identity = (
            db.query(AuthIdentity)
              .filter_by(provider=Provider.GOOGLE, provider_user_id=google_sub)
              .first()
        )

        if identity:
            db_user = identity.user
            identity.last_login_at = dt.datetime.now(dt.timezone.utc)
            db.commit()
            print("[OAUTH] Đăng nhập lại:", db_user.id, email)
        else:
            db_user = User(
                email=email,
                full_name=full_name,
                avatar_url=avatar,
            )
            db.add(db_user)
            db.flush()        
            db.refresh(db_user)

            identity = AuthIdentity(
                user_id=db_user.id,
                provider=Provider.GOOGLE,      
                provider_user_id=google_sub,
            )
            db.add(identity)
            db.commit()
            print("[OAUTH] Tạo user mới:", db_user.id, email)
            

    except Exception as e:
        db.rollback()
        import traceback
        print("[DB ERROR]", repr(e))
        traceback.print_exc()
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