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
