/**
 * Format a date string from Go backend (with microseconds) to a readable format
 * Handles the conversion from Go's 6-digit microseconds to JavaScript's 3-digit milliseconds
 */
export function formatDate(dateString: string): string {
	if (!dateString) return 'Unknown';

	try {
		// Handle Go timestamp format with microseconds (6 digits)
		// Convert to JavaScript format (3 digits for milliseconds)
		let jsDateString = dateString;
		if (dateString.includes('.') && dateString.includes('-')) {
			// Match pattern like "2025-10-14T20:05:06.409647-04:00"
			const match = dateString.match(/^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.(\d{6})(.*)$/);
			if (match) {
				const [, base, microseconds, timezone] = match;
				// Convert 6-digit microseconds to 3-digit milliseconds
				const milliseconds = microseconds.substring(0, 3);
				jsDateString = `${base}.${milliseconds}${timezone}`;
			}
		}

		const parsedDate = new Date(jsDateString);
		if (isNaN(parsedDate.getTime())) {
			console.warn('Invalid date string:', dateString);
			return dateString; // Return original string if invalid
		}
		return parsedDate.toLocaleString();
	} catch (err) {
		console.warn('Error parsing date:', dateString, err);
		return dateString; // Return original string if error
	}
}

/**
 * Format a date string to a relative time (e.g., "2 hours ago")
 */
export function formatRelativeTime(dateString: string): string {
	if (!dateString) return 'Unknown';

	try {
		// Handle Go timestamp format with microseconds (6 digits)
		let jsDateString = dateString;
		if (dateString.includes('.') && dateString.includes('-')) {
			const match = dateString.match(/^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.(\d{6})(.*)$/);
			if (match) {
				const [, base, microseconds, timezone] = match;
				const milliseconds = microseconds.substring(0, 3);
				jsDateString = `${base}.${milliseconds}${timezone}`;
			}
		}

		const date = new Date(jsDateString);
		if (isNaN(date.getTime())) {
			return dateString;
		}

		const now = new Date();
		const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);

		if (diffInSeconds < 60) return `${diffInSeconds}s ago`;
		const diffInMinutes = Math.floor(diffInSeconds / 60);
		if (diffInMinutes < 60) return `${diffInMinutes}m ago`;
		const diffInHours = Math.floor(diffInMinutes / 60);
		if (diffInHours < 24) return `${diffInHours}h ago`;
		const diffInDays = Math.floor(diffInHours / 24);
		return `${diffInDays}d ago`;
	} catch (err) {
		console.warn('Error parsing date for relative time:', dateString, err);
		return dateString;
	}
}
