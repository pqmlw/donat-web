(() => {
  const $ = (s) => document.querySelector(s);
  const CIRC = 2 * Math.PI * 46;

  /* ---------- navigasi ---------- */
  const views = { home: $("#view-home"), tutorial: $("#view-tutorial"), about: $("#view-about") };
  const titles = { home: "Donat Classification", tutorial: "Tutorial · Donat Classification", about: "About · Donat Classification" };

  function route() {
    const h = location.hash.replace(/^#\/?/, "");
    const name = views[h] ? h : "home";
    for (const [k, el] of Object.entries(views)) el.hidden = k !== name;
    document.querySelectorAll(".menu a").forEach((a) => {
      if (a.dataset.nav === name) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
    });
    document.title = titles[name];
    if (name !== "home") stopCamera();
    window.scrollTo(0, 0);
  }
  window.addEventListener("hashchange", route);

  /* ---------- status server ---------- */
  fetch("/api/health").then((r) => r.json()).then((h) => {
    if (!h.ready) showBanner(h.error || "Model belum siap.");
  }).catch(() => showBanner("Server tidak bisa dihubungi. Jalankan app.py lalu muat ulang halaman."));

  function showBanner(msg) { const b = $("#banner"); b.textContent = msg; b.hidden = false; }

  /* ---------- tab upload / camera ---------- */
  const tabUpload = $("#tab-upload"), tabCamera = $("#tab-camera");
  function selectTab(which) {
    const up = which === "upload";
    tabUpload.setAttribute("aria-selected", up);
    tabCamera.setAttribute("aria-selected", !up);
    tabUpload.tabIndex = up ? 0 : -1;
    tabCamera.tabIndex = up ? -1 : 0;
    $("#pane-upload").hidden = !up;
    $("#pane-camera").hidden = up;
    if (up) stopCamera();
  }
  tabUpload.addEventListener("click", () => selectTab("upload"));
  tabCamera.addEventListener("click", () => selectTab("camera"));
  [tabUpload, tabCamera].forEach((t) => t.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      const next = t === tabUpload ? tabCamera : tabUpload;
      next.click(); next.focus();
    }
  }));

  /* ---------- upload ---------- */
  const file = $("#file"), drop = $("#drop");
  $("#btn-upload").addEventListener("click", () => file.click());
  file.addEventListener("change", () => { if (file.files[0]) analyze(file.files[0]); file.value = ""; });
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) analyze(f);
  });

  /* ---------- kamera ---------- */
  const video = $("#video"), camBox = $("#cam-box");
  let stream = null;

  async function startCamera() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 } }, audio: false });
      video.srcObject = stream;
      await video.play();
      camBox.classList.add("live");
      $("#cam-actions").hidden = false;
    } catch (err) {
      $("#cam-msg").querySelector(".hint").textContent =
        err && err.name === "NotAllowedError"
          ? "Izin kamera ditolak. Izinkan kamera di pengaturan browser, lalu coba lagi."
          : "Kamera tidak tersedia. Pakai tab Upload, atau buka lewat localhost / HTTPS.";
    }
  }
  function stopCamera() {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null; video.srcObject = null;
    camBox.classList.remove("live");
    $("#cam-actions").hidden = true;
  }
  $("#btn-cam-start").addEventListener("click", startCamera);
  $("#btn-snap").addEventListener("click", () => {
    const c = $("#snap");
    c.width = video.videoWidth; c.height = video.videoHeight;
    c.getContext("2d").drawImage(video, 0, 0);
    c.toBlob((blob) => { if (blob) analyze(new File([blob], "kamera.jpg", { type: "image/jpeg" })); }, "image/jpeg", 0.92);
  });

  /* ---------- analisis ---------- */
  const cardInput = $("#card-input"), cardResult = $("#card-result");
  const info = $("#info"), fail = $("#fail"), busy = $("#busy");
  let objectUrl = null;

  function reset() {
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = null;
    cardResult.hidden = true; cardInput.hidden = false;
    info.hidden = true; fail.hidden = true; $("#shot").hidden = false;
    $(".more").open = false;
  }
  $("#btn-again").addEventListener("click", reset);
  $("#btn-retry").addEventListener("click", reset);
  $(".brand").addEventListener("click", () => { if (location.hash === "" || location.hash === "#/") reset(); });

  async function analyze(f) {
    stopCamera();
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(f);
    $("#res-img").src = objectUrl;
    cardInput.hidden = true; cardResult.hidden = false;
    info.hidden = true; fail.hidden = true; busy.hidden = false;
    cardResult.classList.add("loading");

    const fd = new FormData();
    fd.append("image", f);
    try {
      const r = await fetch("/api/predict", { method: "POST", body: fd });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.error || "Terjadi kesalahan di server.");
      render(data);
    } catch (err) {
      $("#fail-msg").textContent = err.message === "Failed to fetch"
        ? "Server tidak bisa dihubungi. Pastikan app.py masih berjalan."
        : err.message;
      $("#shot").hidden = true;      // foto tidak valid: jangan tampilkan gambar rusak
      fail.hidden = false;
    } finally {
      busy.hidden = true;
      cardResult.classList.remove("loading");
    }
  }

  function render(d) {
    const low = d.confidence === "LOW";
    info.className = "info " + (d.passed ? "pass" : "fail") + (low ? " low" : "");
    $("#res-kind").textContent = d.kind;
    $("#res-status").textContent = d.status || (d.passed ? "Lolos" : "Tidak lolos");
    $("#res-pct").textContent = d.percentage.toFixed(0) + "%";
    const conf = $("#res-conf");
    conf.textContent = d.confidence; conf.className = d.confidence;

    const note = $("#res-note");
    note.hidden = !low;
    if (low) note.textContent = "Model ragu dengan foto ini. Periksa donat secara manual atau coba foto lain.";

    $("#g-icon").innerHTML = d.passed
      ? '<path d="M40 62 L54 76 L82 44"/>'
      : '<path d="M44 44 L76 76 M76 44 L44 76"/>';

    const bar = $("#g-bar");
    bar.style.strokeDasharray = `0 ${CIRC}`;
    requestAnimationFrame(() => requestAnimationFrame(() => {
      bar.style.strokeDasharray = `${CIRC * d.probability} ${CIRC}`;
    }));

    const list = $("#rank"); list.textContent = "";
    d.ranking.forEach((r) => {
      const li = document.createElement("li");
      const name = document.createElement("span");
      name.textContent = `${r.kind}${r.status ? ", " + r.status.toLowerCase() : ""}`;
      const pct = document.createElement("span");
      pct.textContent = r.percentage.toFixed(1) + "%";
      const bar2 = document.createElement("span"); bar2.className = "bar";
      const fill = document.createElement("i"); fill.style.width = r.percentage + "%";
      bar2.append(fill);
      li.append(name, pct, bar2);
      list.append(li);
    });
    $("#res-proc").src = d.processed;
    info.hidden = false;
  }

  route();
})();
