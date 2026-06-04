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
   full-screen. The brand is a link to index.html, so a click normally
   navigates/reloads the page and would wipe an in-memory counter; we keep
   the click timestamps in sessionStorage so they survive that navigation,
   and we check the count both on each click and on page load. Click
   anywhere (or press Esc) to dismiss. */
(function () {
  var NEEDED = 5;      // clicks required
  var WINDOW = 2500;   // …within this many ms (must span a page reload)
  var STORE = "td-egg-clicks";
  var overlay = null;

  function recent() {
    var now = Date.now(), arr = [];
    try { arr = JSON.parse(sessionStorage.getItem(STORE)) || []; } catch (e) {}
    return arr.filter(function (t) { return now - t < WINDOW; });
  }
  function save(arr) { try { sessionStorage.setItem(STORE, JSON.stringify(arr)); } catch (e) {} }
  function clear() { try { sessionStorage.removeItem(STORE); } catch (e) {} }

  function onKey(e) { if (e.key === "Escape") hide(); }
  function hide() {
    if (overlay) { overlay.remove(); overlay = null; }
    document.removeEventListener("keydown", onKey);
  }

  function reveal() {
    if (overlay) return;
    overlay = document.createElement("div");
    overlay.style.cssText =
      "position:fixed;inset:0;z-index:2147483647;background:#000;" +
      "display:flex;align-items:center;justify-content:center;cursor:pointer;";
    var img = document.createElement("img");
    img.src = "visualizations/ian.png";
    img.alt = "ian";
    img.style.cssText = "width:100%;height:100%;object-fit:contain;";
    overlay.appendChild(img);
    overlay.addEventListener("click", hide);
    document.addEventListener("keydown", onKey);
    document.body.appendChild(overlay);
  }

  document.addEventListener("DOMContentLoaded", function () {
    // A click on a previous page may have just put us over the threshold.
    if (recent().length >= NEEDED) { clear(); reveal(); }

    var brand = document.querySelector(".brand");
    if (!brand) return;
    brand.addEventListener("click", function (e) {
      var arr = recent();
      arr.push(Date.now());
      save(arr);
      if (arr.length >= NEEDED) {
        e.preventDefault(); // stay on the page and show the image now
        clear();
        reveal();
      }
      // otherwise let the link navigate; the counter persists across the load
    });
  });
})();
