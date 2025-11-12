import { Clock, Plus, Sparkles } from "lucide-react";
import { useState } from "react";
import "../../styles/HistoryPanel.css";

export type HistoryItem = {
  id: number | string;
  title: string;
  time: string;
};

type Props = {
  items: HistoryItem[];
  className?: string;
  onItemClick?: (id: number | string) => void;
  activeId?: number | string;                   
};

export default function HistoryPanel({ items, className, onItemClick, activeId }: Props) {
  const [hoveredId, setHoveredId] = useState<number | string | null>(null);

  return (
    <div className={`hp-root ${className || ""}`}>

      <div className="hp-header">
        <div className="hp-header-bg" />
        <div className="hp-header-row">
          <div className="hp-title-wrap">
            <div className="hp-title-icon">
              <Clock className="h-5 w-5 text-blue-600 dark:text-blue-400" />
              <div className="hp-ping" />
            </div>
            <h2 className="hp-title-text">Lịch sử</h2>
          </div>

          <button
            onClick={() => (window.location.href = "/")}
            className="group hp-new-btn"
            aria-label="Tóm tắt mới"
            title="Tóm tắt mới"
          >
            <div className="hp-new-btn-hover" />
            <Plus className="relative h-4 w-4 text-white" />
          </button>
        </div>
      </div>

      <div className="hp-body" role="listbox" aria-label="Danh sách lịch sử"
           aria-activedescendant={activeId ? `hp-item-${activeId}` : undefined}>
        {items.length === 0 ? (
          <div className="hp-empty">
            <div className="hp-empty-aura">
              <Clock className="relative h-16 w-16 text-slate-300 dark:text-slate-700" />
            </div>
            <p className="hp-empty-title">Chưa có lịch sử nào</p>
            <p className="hp-empty-sub">Bắt đầu tạo tóm tắt đầu tiên của bạn</p>
          </div>
        ) : (
          <div className="hp-list" role="presentation">
            {items.map((item, index) => {
              const isActive = activeId === item.id;
              return (
                <div
                  key={item.id}
                  id={`hp-item-${item.id}`}
                  onMouseEnter={() => setHoveredId(item.id)}
                  onMouseLeave={() => setHoveredId(null)}
                  className={`group hp-item ${isActive ? "ring-1 ring-blue-400/50" : ""}`}
                  style={{ animationDelay: `${index * 50}ms` }}
                  role="option"
                  aria-selected={isActive}
                >
                  <div className="hp-item-outer" />
                  <div className="hp-item-inner" />

                  <button
                    className="hp-item-content"
                    title={item.title}
                    onClick={() => onItemClick?.(item.id)}               
                    onKeyDown={(e) => {                                    
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        onItemClick?.(item.id);
                      }
                    }}
                    aria-current={isActive ? "true" : undefined}
                  >
                    <div className="hp-item-icon-wrap">
                      <div className={`hp-item-icon ${hoveredId === item.id ? "hp-item-icon--active" : ""}`}>
                        <Sparkles
                          className={`h-4 w-4 ${hoveredId === item.id ? "text-white" : "text-slate-500 dark:text-slate-400"}`}
                        />
                      </div>
                    </div>

                    <div className="hp-item-text">
                      <h3 className="hp-item-title">{item.title}</h3>
                      <div className="hp-item-time">
                        <Clock className="h-3.5 w-3.5 text-slate-400 dark:text-slate-500" />
                        <p className="hp-item-time-text">{item.time}</p>
                      </div>
                    </div>

                    <div className={`hp-item-arrow ${hoveredId === item.id ? "hp-item-arrow--show" : ""}`}>
                      <div className="hp-item-arrow-pill">
                        <svg className="h-3 w-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                      </div>
                    </div>
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {items.length > 0 && (
        <div className="hp-footer">
          <div className="hp-footer-bg" />
          <div className="hp-footer-row">
            <div className="hp-badge">{items.length}</div>
            <p className="hp-footer-text">tóm tắt đã lưu</p>
          </div>
        </div>
      )}
    </div>
  );
}
