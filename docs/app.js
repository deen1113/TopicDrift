/* TopicDrift — shared site behaviour.
   Site-wide dark mode that (a) persists across pages via localStorage and
   (b) is also carried on the navigation links themselves, so the choice
   survives page-to-page navigation even when storage is unavailable
   (e.g. pages opened directly over file://). It then drives the embedded
   figures regardless of iframe load timing. */
(function () {
  var root = document.documentElement;
  var KEY = "td-dark";

  // ── preference storage (degrades gracefully) ──────────────────────────────
  function readStore() { try { return localStorage.getItem(KEY); } catch (e) { return null; } }
  function writeStore(on) { try { localStorage.setItem(KEY, on ? "1" : "0"); } catch (e) {} }

  // The theme can also ride along in the URL hash (#td=dark / #td=light),
  // which is how navigation links pass it when storage can't.
  function prefFromHash() {
    var h = location.hash || "";
    if (h.indexOf("td=dark") >= 0) return true;
    if (h.indexOf("td=light") >= 0) return false;
    return null;
  }
  function initialPref() {
    var h = prefFromHash();
    return h !== null ? h : readStore() === "1";
  }

  function isDark() { return root.classList.contains("dark"); }

  // ── carry the theme on internal links ─────────────────────────────────────
  function tagLinks() {
    var frag = "#td=" + (isDark() ? "dark" : "light");
    document.querySelectorAll('a[href$=".html"]').forEach(function (a) {
      var href = a.getAttribute("href");
      if (!href || /^(https?:)?\/\//.test(href)) return; // skip external links
      a.setAttribute("href", href.split("#")[0] + frag);
    });
  }

  // ── drive embedded figures ────────────────────────────────────────────────
  function postTo(frame) {
    try { frame.contentWindow.postMessage({ type: "td-dark", on: isDark() }, "*"); } catch (e) {}
  }
  function syncFrames() {
    document.querySelectorAll("iframe").forEach(function (f) {
      if (!f.dataset.tdBound) {
        f.dataset.tdBound = "1";
        f.addEventListener("load", function () { postTo(f); });
      }
      postTo(f);
    });
  }

  function apply(on) {
    root.classList.toggle("dark", on);
    var cb = document.getElementById("dm-toggle");
    if (cb) cb.checked = on;
    syncFrames();
    tagLinks();
  }

  // Apply as early as possible (the inline <head> script already set the class
  // to avoid a flash; this keeps the checkbox, links and figures in step).
  apply(initialPref());

  document.addEventListener("DOMContentLoaded", function () {
    var cb = document.getElementById("dm-toggle");
    if (cb) {
      cb.checked = isDark();
      cb.addEventListener("change", function () {
        writeStore(cb.checked);
        apply(cb.checked);
      });
    }
    apply(isDark()); // re-tag links now the whole DOM is parsed
  });

  // Safety net once everything (including iframes) has fully loaded.
  window.addEventListener("load", syncFrames);
})();

/* Easter egg — five fast clicks on the TopicDrift brand reveals ian.png
   full-screen. Click anywhere (or press Esc) to dismiss. */
(function () {
  var NEEDED = 5;      // clicks required
  var WINDOW = 1500;   // …within this many ms
  var clicks = [];
  var overlay = null;

  function close() {
    if (overlay) { overlay.remove(); overlay = null; }
    document.removeEventListener("keydown", onKey);
  }

  function onKey(e) { if (e.key === "Escape") close(); }

  function reveal() {
    if (overlay) return;
    overlay = document.createElement("div");
    overlay.style.cssText =
      "position:fixed;inset:0;z-index:99999;background:rgba(0,0,0,0.92);" +
      "display:flex;align-items:center;justify-content:center;cursor:pointer;";
    var img = document.createElement("img");
    img.src = "visualizations/ian.png";
    img.alt = "ian";
    img.style.cssText = "max-width:100vw;max-height:100vh;object-fit:contain;";
    overlay.appendChild(img);
    overlay.addEventListener("click", close);
    document.addEventListener("keydown", onKey);
    document.body.appendChild(overlay);
  }

  document.addEventListener("DOMContentLoaded", function () {
    var brand = document.querySelector(".brand");
    if (!brand) return;
    brand.addEventListener("click", function (e) {
      var now = Date.now();
      clicks.push(now);
      clicks = clicks.filter(function (t) { return now - t < WINDOW; });
      if (clicks.length >= NEEDED) {
        e.preventDefault(); // suppress navigation on the rapid final click
        clicks = [];
        reveal();
      }
    });
  });
})();
