export function firstWords(text?: string | null, n = 5, maxLen = 10) {
  if (!text) return "Không có tiêu đề";
  const clean = text.replace(/\s+/g, " ").trim();
  if (!clean) return "Không có tiêu đề";

  const words = clean.split(" ").slice(0, n).join(" ");
  let title = words;
  if (clean.length > words.length) title += "…"; 
  if (title.length > maxLen) title = title.slice(0, maxLen).trimEnd() + "…";
  return title;
}
