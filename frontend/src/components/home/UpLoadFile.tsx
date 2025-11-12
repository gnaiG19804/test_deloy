import { FileText } from "lucide-react";
import "../../styles/UpLoadFile.css";
import * as mammoth from "mammoth";

interface UpLoadFileProps {
  onFileLoaded: (text: string) => void;
  onError?: (message: string) => void;
  disabled?: boolean;
}

export default function UpLoadFile({
  onFileLoaded,
  onError,
  disabled = false,
}: UpLoadFileProps) {
  const fail = (msg: string, input?: HTMLInputElement) => {
    if (onError) onError(msg);
    else alert(msg); 
    if (input) input.value = "";
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (disabled) return;
    const file = e.target.files?.[0];
    if (!file) return;

    const ext = file.name.split(".").pop()?.toLowerCase();

    try {
      let content = "";

      if (ext === "txt") {
        content = await file.text();
      } else if (ext === "docx") {
        const arrayBuffer = await file.arrayBuffer();
        const { value } = await mammoth.extractRawText({ arrayBuffer });
        content = value;
      } else {
        fail("❌ Sai định dạng! Chỉ hỗ trợ tệp .txt hoặc .docx.", e.target);
        return;
      }

      if (!content.trim()) {
        fail("⚠️ Tệp trống! Vui lòng chọn tệp có nội dung.", e.target);
        return;
      }

      onFileLoaded(content.trim());
    } catch (err) {
      console.error("Lỗi khi đọc file:", err);
      fail("❌ Không thể đọc nội dung tệp.", e.target);
    }

    e.target.value = "";
  };

  return (
    <label
      className={`ip-upload flex items-center gap-2 px-4 py-2 rounded-lg border
        cursor-pointer transition-all duration-200
        ${disabled ? "opacity-60 cursor-not-allowed" : "hover:bg-indigo-50"}`}
      aria-disabled={disabled}
    >
      <FileText className="ip-icon" />
      <span>Tải lên tệp tin (.txt / .docx)</span>
      <input
        type="file"
        accept=".txt,.docx"
        className="hidden"
        onChange={handleFileChange}
        disabled={disabled}
      />
    </label>
  );
}
