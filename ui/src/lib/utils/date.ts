// Go timestamps carry up to nine fractional digits; Date only reads three
export function formatDate(s: string): string {
	if (!s) return '';
	const d = new Date(s.replace(/(\.\d{3})\d+/, '$1'));
	return isNaN(d.getTime()) ? s : d.toLocaleString();
}
