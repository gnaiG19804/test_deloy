export function toViRelative(dateIso: string) {
  const d = new Date(dateIso);
  const now = new Date();
  const isSameDay = d.toDateString() === now.toDateString();

  const yesterday = new Date(now);
  yesterday.setDate(now.getDate() - 1);
  const isYesterday = d.toDateString() === yesterday.toDateString();

  if (isSameDay) return "Hôm nay";
  if (isYesterday) return "Hôm qua";

  const dd = String(d.getDate()).padStart(2, "0");
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  return `${dd}/${mm}`;
}
