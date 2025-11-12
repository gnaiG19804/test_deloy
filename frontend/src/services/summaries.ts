
export type SummaryView = {
  id: string;
  original_text: string | null;  
  summary_text: string;
  created_at: string;          
};

export type PredictIn = { text: string };

const API_BASE = (import.meta.env.VITE_API_BASE ?? "http://localhost:8000").replace(/\/+$/,"");
const SUMMARIES_URL = `${API_BASE}/summaries`;

function authHeaders(token?: string) {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function parseOrThrow(res: Response) {
  let payload: any = null;
  try { payload = await res.json(); } catch {}
  if (!res.ok) {
    const msg = (payload && (payload.detail || payload.message)) || `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return payload;
}

export async function getSummaries(skip = 0, limit = 20, token?: string) {
  const res = await fetch(`${SUMMARIES_URL}?skip=${skip}&limit=${limit}`, {
    method: "GET",
    headers: {
      "Accept": "application/json",
      ...authHeaders(token),
    },
    credentials: "include",
  });
  return parseOrThrow(res) as Promise<SummaryView[]>;
}

export async function getSummaryById(id: string, token?: string): Promise<SummaryView> {
  const res = await fetch(`${SUMMARIES_URL}/${id}`, {
    method: "GET",
    headers: {
      "Accept": "application/json",
      ...authHeaders(token),
    },
    credentials: "include",
  });
  return parseOrThrow(res) as Promise<SummaryView>;
}

export async function predictSummary(
  input: PredictIn,
  token?: string,
  opts?: { timeoutMs?: number }
): Promise<SummaryView> {
  const text = input?.text?.trim();
  if (!text) throw new Error("Văn bản trống");

  const timeoutMs = opts?.timeoutMs ?? 15000;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);

  try {
    const res = await fetch(
      `${SUMMARIES_URL}/predict`,
      {
        method: "POST",
        headers: {
          "Accept": "application/json",
          "Content-Type": "application/json",
          ...authHeaders(token),
        },
        body: JSON.stringify({ text }),
        credentials: "include",
        signal: ctrl.signal,
      }
    );
    return await parseOrThrow(res) as SummaryView;
  } catch (e: any) {
    if (e?.name === "AbortError") {
      throw new Error("Hết thời gian chờ phản hồi (timeout). Thử lại sau.");
    }
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

// loại bỏ khoảng trắng thừa trong văn tải lên và nhập liệu
export function cleanInputText(text: string): string {
  return text.replace(/\s+/g, ' ').trim();
}

const EMOJI_RE =
  /[\p{Extended_Pictographic}\uFE0F\u200D\u{1F1E6}-\u{1F1FF}\u{1F3FB}-\u{1F3FF}]/gu;
const INVISIBLES_RE = /[\u200B-\u200D\u2060\u00A0\uFEFF]/g;

export function hasEmoji(s: string) {
  return EMOJI_RE.test(s);
}

//  xử lý văn bản nếu có emoji xóa bỏ emoji
export function removeEmojis(text: string): string {
  return text.replace(EMOJI_RE, '').replace(INVISIBLES_RE, '');
}