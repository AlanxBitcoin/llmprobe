(() => {
  const parts = [
    "/static/app.homepage.attribute_groups.js",
    "/static/app.homepage.awr.js",
    "/static/app.homepage.js",
    "/static/app.homepage.chat.js",
  ];

  const loadPart = (index) => {
    if (index >= parts.length) return;
    const script = document.createElement("script");
    script.src = parts[index];
    script.async = false;
    script.onload = () => loadPart(index + 1);
    script.onerror = () => {
      const msg = `Failed to load ${parts[index]}`;
      console.error(msg);
      const status = document.getElementById("serverStatus");
      if (status) status.textContent = "Error";
      const summary = document.getElementById("resultSummary");
      if (summary) summary.textContent = msg;
    };
    document.body.appendChild(script);
  };

  loadPart(0);
})();
