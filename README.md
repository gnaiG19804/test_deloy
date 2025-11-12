# 🧠 Web Tóm Tắt Văn Bản


## 🧩 Yêu cầu hệ thống
- Python 3.11.x
- Node.js ≥ 18.x
- npm ≥ 9.x


---

## ⚙️ Cài đặt tự động (Windows)

Sử dụng script `setup.ps1` (khuyến nghị):
1. Mở PowerShell với quyền thường (không cần Administrator) đến thư mục gốc của dự án.
2. Chạy lệnh:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\setup.ps1
```

Script sẽ:
- Kiểm tra và cài Node.js nếu chưa tồn tại.
- Tạo và kích hoạt virtualenv cho backend.
- Cài đặt toàn bộ dependencies Python và npm.

---

## 🛠 Cài đặt thủ công

### 1 Backend 

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2 Frontend 

```bash
cd frontend
npm install
```
 
## 3 Khởi động chương trình

### Backend
```powershell
cd backend
uvicorn app.main:app --reload --port 8000
```
Backend mặc định phục vụ tại `http://localhost:8000`.

### Frontend

```powershell
cd frontend
npm run dev
```
Frontend mặc định chạy tại `http://localhost:5173`.



