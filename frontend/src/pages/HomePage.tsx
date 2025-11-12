import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { PanelLeftOpen, X } from "lucide-react";
import Header from "../components/home/Header";
import InputPanel from "../components/home/InputPanel";
import HistoryPanel, { HistoryItem } from "../components/home/HistoryPanel";
import { getSummaries, getSummaryById, SummaryView } from "../services/summaries";
import { toViRelative } from "../utils/relativeTime";
import { firstWords } from "../utils/firstWords";
import "../styles/HomePage.css";

export default function HomePage() {
  const [openHistory, setOpenHistory] = useState(false);
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const [currentSummary, setCurrentSummary] = useState<SummaryView | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const navigate = useNavigate();
  const { id } = useParams<{ id: string }>();

  useEffect(() => {
    (async () => {
      try {
        setLoading(true);
        const token = localStorage.getItem("token") ?? undefined;

        if (!token) {
          setItems([]);
          setLoading(false);
        }

        const data: SummaryView[] = await getSummaries(0, 20, token);
        const mapped: HistoryItem[] = data.map((s) => ({
          id: s.id,
          title: s.original_text
            ? firstWords(s.original_text, 10, 70)
            : "(Không có văn bản gốc)",
          time: toViRelative(s.created_at),
        }));

        setItems(mapped);
      } catch (e: any) {
        if (String(e.message).includes("401")) {
          setItems([]);
        } else {
          console.error("Lỗi tải lịch sử:", e);
        }
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  useEffect(() => {
    if (!id) {
      setCurrentSummary(null);
      return;
    }
    (async () => {
      try {
        setDetailLoading(true);
        const token = localStorage.getItem("token") ?? undefined;
        const data = await getSummaryById(id, token);
        setCurrentSummary(data);
      } catch (e: any) {
        console.error("Lỗi tải chi tiết:", e);
        setCurrentSummary(null);
      } finally {
        setDetailLoading(false);
      }
    })();
  }, [id]);

  return (
    <div>
      <Header />

      <button
        onClick={() => setOpenHistory(true)}
        className="hp-history-btn"
        aria-label="Mở lịch sử"
      >
        <PanelLeftOpen className="h-4 w-4" />
        <span>Lịch sử</span>
      </button>

      <aside
        className={`hp-sidebar ${openHistory ? "translate-x-0" : "-translate-x-full"}`}
        role="dialog"
        aria-modal="true"
      >
        <div className="hp-sidebar-header">
          <button
            onClick={() => setOpenHistory(false)}
            className="hp-close-btn"
            aria-label="Đóng"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="hp-sidebar-content">
          {loading ? (
            <div className="hp-status-text">Đang tải…</div>
          ) : err ? (
            <div className="hp-status-error">Lỗi: {err}</div>
          ) : (
            <HistoryPanel
              items={items}
              activeId={id}
              onItemClick={(sid) => {
                setOpenHistory(false);
                navigate(`/summaries/${sid}`);
              }}
              className="w-full p-0 bg-transparent shadow-none"
            />
          )}
        </div>
      </aside>

      <main>
        <InputPanel
          originalText={currentSummary?.original_text || ""}
          summaryText={currentSummary?.summary_text || ""}
          readOnly={!!currentSummary}
        />

        {detailLoading && (
          <div style={{ textAlign: "center", marginTop: "1rem", color: "#666" }}>
            Đang tải chi tiết…
          </div>
        )}
      </main>
    </div>
  );
}
