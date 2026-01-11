const http = require("node:http");
const https = require("node:https");
const { URL } = require("node:url");

function requestJson(url, payload) {
  return new Promise((resolve, reject) => {
    const parsed = new URL(url);
    const body = JSON.stringify(payload);
    const options = {
      method: "POST",
      hostname: parsed.hostname,
      port: parsed.port || (parsed.protocol === "https:" ? 443 : 80),
      path: parsed.pathname + parsed.search,
      headers: {
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(body),
      },
    };
    const client = parsed.protocol === "https:" ? https : http;
    const req = client.request(options, (resp) => {
      let data = "";
      resp.on("data", (chunk) => (data += chunk));
      resp.on("end", () => {
        let parsedBody = null;
        try {
          parsedBody = data ? JSON.parse(data) : null;
        } catch {
          parsedBody = null;
        }
        resolve({ status: resp.statusCode || 500, data: parsedBody });
      });
    });
    req.on("error", reject);
    req.write(body);
    req.end();
  });
}

module.exports = async (req, res) => {
  try {
    const host = req.headers["x-forwarded-host"] || req.headers.host || "";
    const origin = host ? `https://${host}` : "https://medibot-ready.vercel.app";
    const backend = process.env.BACKEND_TUNNEL_URL || "";

    const query = req.query || {};
    const getParam = (value) => (Array.isArray(value) ? value[0] : value);
    const code = getParam(query.code);
    const state = getParam(query.state);
    const error = getParam(query.error);

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

    let result;
    try {
      const url = `${backend.replace(/\\/+$/, "")}/api/auth/google/complete`;
      result = await requestJson(url, { code, state });
    } catch (err) {
      res.status(502).send("Failed to reach backend for OAuth completion.");
      return;
    }

    if (!result || result.status >= 400) {
      res.status(result?.status || 500).send(result?.data?.detail || "OAuth completion failed.");
      return;
    }

    const data = result.data || {};
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
  } catch (err) {
    res.status(500).send("OAuth callback failed.");
  }
};
