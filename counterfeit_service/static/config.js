/**
 * Единственная точка настройки URL бэкенда для демо-страницы.
 *
 * Локально (README: uvicorn --port 8000): значение по умолчанию.
 *
 * GitHub Pages / любой статический хост: замените на полный HTTPS URL API,
 * например выход Cloudflare Tunnel (`https://xxxx.trycloudflare.com`)
 * или прод-домен облака — иначе fetch уйдёт не туда (CORS / 404).
 */
window.API_BASE_URL = "http://localhost:8000";
window.PREDICT_ENDPOINT = `${String(window.API_BASE_URL).replace(/\/$/, "")}/predict`;
