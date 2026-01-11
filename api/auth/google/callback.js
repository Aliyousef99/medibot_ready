module.exports = async (req, res) => {
  const host = req.headers["x-forwarded-host"] || req.headers.host || "";
  const origin = host ? `https://${host}` : "https://medibot-ready.vercel.app";
  const backend = process.env.BACKEND_TUNNEL_URL || "";

  const getParam = (value) => (Array.isArray(value) ? value[0] : value);
  const code = getParam(req.query.code);
  const state = getParam(req.query.state);
  const error = getParam(req.query.error);

  if (error) {
    res.status(400).send(`OAuth error: ${error}`);
    return;
  }
  if (!backend) {
    res.status(500).send("Backend tunnel URL is not configured.");
    return;
  }
  if (!code || !state) {
    res.status(400).send("Missing OAuth parameters.");
    return;
  }

  let data;
  try {
    const response = await fetch(`${backend.replace(/\\/+$/, "")}/api/auth/google/complete`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code, state }),
    });
    data = await response.json();
    if (!response.ok) {
      res.status(response.status).send(data?.detail || "OAuth completion failed.");
      return;
    }
  } catch (err) {
    res.status(502).send("Failed to reach backend for OAuth completion.");
    return;
  }

  const redirect = typeof data.redirect === "string" && data.redirect.startsWith(origin)
    ? data.redirect
    : `${origin}/#/`;

  const payload = {
    email: data.email || "",
    token: data.access_token || "",
    refresh_token: data.refresh_token || "",
  };

  res.setHeader("Content-Type", "text/html; charset=utf-8");
  res.status(200).send(`<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Signing in...</title>
  </head>
  <body>
    <script>
      try {
        var auth = ${JSON.stringify(payload)};
        localStorage.setItem("auth", JSON.stringify(auth));
      } catch (e) {}
      window.location.replace(${JSON.stringify(redirect)});
    </script>
  </body>
</html>`);
};
