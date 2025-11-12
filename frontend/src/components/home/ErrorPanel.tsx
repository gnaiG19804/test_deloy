import { X, Sparkles } from "lucide-react";
import { useEffect, useRef } from "react";
import "../../styles/ErrorPanel.css";

type ErrorPanelProps = {
  open: boolean;
  title?: string;
  message?: string | React.ReactNode;
  onClose: () => void;
  confirmText?: string;
  onConfirm?: () => void;
};

export default function ErrorPanel({
  open,
  title = "Thông báo",
  message,
  onClose,
  confirmText,
  onConfirm,
}: ErrorPanelProps) {
  const panelRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="error-overlay"
      aria-modal="true"
      role="dialog"
      aria-labelledby="modal-title"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      {/* Animated Background */}
      <div className="error-bg-gradient animate-gradient" />
      <div className="error-bg-dark" />

      {/* Floating particles */}
      <div className="error-particles">
        <div className="particle particle-1" />
        <div className="particle particle-2" />
        <div className="particle particle-3" />
        <div className="particle particle-4" />
      </div>

      {/* Panel */}
      <div ref={panelRef} className="error-panel modal-enter">
        <div className="error-glow animate-pulse-slow" />

        <div className="error-box">
          <div className="error-top-border animate-shimmer" />

          <div className="error-corner" />

          {/* Header */}
          <div className="error-header">
            <div className="error-header-content">
              <div className="error-icon">
                <Sparkles className="w-5 h-5 text-white" />
              </div>
              <h3 id="modal-title" className="error-title">
                {title}
              </h3>
            </div>
            <button
              onClick={onClose}
              className="error-close-btn"
              aria-label="Đóng"
            >
              <X className="w-5 h-5 text-slate-400 group-hover:text-slate-700 transition-colors" />
            </button>
          </div>

          {/* Body */}
          <div className="error-body">
            {typeof message === "string" ? <p>{message}</p> : message}
          </div>

          {/* Footer */}
          <div className="error-footer">
            <div className="error-footer-accent" />
            <div className="error-footer-buttons">
              <button onClick={onClose} className="btn-close">
                Đóng
              </button>

              {confirmText && (
                <button onClick={onConfirm} className="btn-confirm">
                  {confirmText}
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
