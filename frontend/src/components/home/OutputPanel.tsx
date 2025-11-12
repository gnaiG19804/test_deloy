import "../../styles/OutputPanel.css";

export default function OutputPanel({ text }: { text?: string }) {
  const value = (text ?? "").trim();

  return (
    <div className="ip-output-area">
      {value ? (
        <textarea
          className="ip-output-textarea"
          value={value}
          readOnly
        />
      ) : (
        <p className="ip-output-placeholder">
          Kết quả tóm tắt sẽ hiển thị ở đây...
        </p>
      )}
    </div>
  );
}
