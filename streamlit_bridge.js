(() => {
  const post = (type, extra) =>
    window.parent.postMessage(Object.assign({ isStreamlitMessage: true, type }, extra || {}), "*");

  let args = null;               
  let firstRender;
  const gotFirst = new Promise((r) => { firstRender = r; });
  let pending = null;             

  window.addEventListener("message", (e) => {
    const d = e.data;
    if (!d || d.type !== "streamlit:render") return;
    args = d.args || {};
    firstRender();
    const res = args.result;
    if (pending && res && res.nonce === pending.nonce) {
      clearTimeout(pending.timer);
      const p = pending; pending = null;
      p.resolve(res);
    }
  });

  post("streamlit:componentReady", { apiVersion: 1 });
  post("streamlit:setFrameHeight", { height: 700 });

  const reply = (data, status) =>
    new Response(JSON.stringify(data), { status, headers: { "Content-Type": "application/json" } });

  async function toDataUrl(file) {
    const bmp = await createImageBitmap(file);
    const MAX = 1600;
    const k = Math.min(1, MAX / Math.max(bmp.width, bmp.height));
    const c = document.createElement("canvas");
    c.width = Math.round(bmp.width * k); c.height = Math.round(bmp.height * k);
    c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
    return c.toDataURL("image/jpeg", 0.92);
  }

  async function predict(file) {
    await gotFirst;
    if (args.ready === false) return reply({ error: args.error || "Model belum siap." }, 503);
    let image;
    try { image = await toDataUrl(file); }
    catch (_) { return reply({ error: "File tidak bisa dibaca sebagai gambar. Gunakan JPG, PNG, atau WEBP." }, 400); }

    const nonce = Date.now() + "-" + Math.random().toString(36).slice(2);
    const result = await new Promise((resolve, reject) => {
      const timer = setTimeout(() => { pending = null; reject(new Error("Waktu habis. Coba lagi.")); }, 180000);
      pending = { nonce, resolve, reject, timer };
      post("streamlit:setComponentValue", { value: { nonce, image }, dataType: "json" });
    }).catch((err) => ({ error: err.message }));
    return reply(result, result.error ? 422 : 200);
  }

  const realFetch = window.fetch.bind(window);
  window.fetch = async (input, init) => {
    const url = typeof input === "string" ? input : (input && input.url) || "";
    if (url.endsWith("/api/health")) {
      await gotFirst;
      return reply({ ready: args.ready !== false, error: args.error || null }, 200);
    }
    if (url.endsWith("/api/predict")) {
      const file = init && init.body && init.body.get && init.body.get("image");
      if (!file) return reply({ error: "Tidak ada gambar yang dikirim." }, 400);
      return predict(file);
    }
    return realFetch(input, init);
  };
})();
