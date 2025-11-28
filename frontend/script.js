const API_URL = "http://127.0.0.1:8000/search";

document.getElementById("searchBtn").addEventListener("click", async () => {
  const query = document.getElementById("query").value.trim();
  if (!query) return;

  const resultsDiv = document.getElementById("results");
  resultsDiv.innerHTML = "<p>🔍 Searching...</p>";

  try {
    const response = await fetch(`${API_URL}?prompt=${encodeURIComponent(query)}`);
    const data = await response.json();
    resultsDiv.innerHTML = "";

    data.results.forEach((res) => {
      const img = document.createElement("img");
      img.src = `http://127.0.0.1:8000${res.url}`;
      img.alt = res.path;
      img.title = `${res.score.toFixed(3)}`;
      resultsDiv.appendChild(img);
    });
  } catch (err) {
    resultsDiv.innerHTML = `<p style="color:red;">Error: ${err.message}</p>`;
  }
});
