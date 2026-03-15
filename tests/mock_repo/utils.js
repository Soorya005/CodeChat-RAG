/**
 * tests/mock_repo/utils.js
 * Helper utilities used as test data for the RAG pipeline (JavaScript).
 */

/**
 * Debounce a function so it is only called after `delay` ms of inactivity.
 * @param {Function} fn    - The function to debounce.
 * @param {number}   delay - Debounce delay in milliseconds.
 * @returns {Function} Debounced wrapper.
 */
function debounce(fn, delay) {
    let timer;
    return function (...args) {
        clearTimeout(timer);
        timer = setTimeout(() => fn.apply(this, args), delay);
    };
}

/**
 * Deep-clone a plain JSON-serialisable object.
 * @param {Object} obj - The object to clone.
 * @returns {Object} A deep copy.
 */
function deepClone(obj) {
    return JSON.parse(JSON.stringify(obj));
}

/**
 * Format a Date object as "YYYY-MM-DD".
 * @param {Date} date
 * @returns {string}
 */
function formatDate(date) {
    const y = date.getFullYear();
    const m = String(date.getMonth() + 1).padStart(2, "0");
    const d = String(date.getDate()).padStart(2, "0");
    return `${y}-${m}-${d}`;
}

module.exports = { debounce, deepClone, formatDate };
