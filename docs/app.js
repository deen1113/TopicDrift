/* TopicDrift — shared site behaviour.
   Site-wide dark mode that persists across pages and reliably drives the
   embedded figures, regardless of iframe load timing. */
(function () {
  var root = document.documentElement;
  var KEY = "td-dark";

  function isDark() { return root.classList.contains("dark"); }

  // Tell one iframe about the current theme (ignored if not ready yet).
  function postTo(frame) {
    try { frame.contentWindow.postMessage({ type: "td-dark", on: isDark() }, "*"); } catch (e) {}
  }

  // Push the current theme into every figure. Already-loaded frames get it
  // immediately; the load listener covers frames still fetching (incl. lazy
  // ones below the fold) so none are missed by a timing race.
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
  }

  // Apply persisted preference as early as possible (the inline <head> script
  // already set the class to avoid a flash; this keeps everything in step).
  apply(localStorage.getItem(KEY) === "1");

  document.addEventListener("DOMContentLoaded", function () {
    var cb = document.getElementById("dm-toggle");
    if (cb) {
      cb.checked = isDark();
      cb.addEventListener("change", function () {
        localStorage.setItem(KEY, cb.checked ? "1" : "0");
        apply(cb.checked);
      });
    }
    syncFrames();
  });

  // Final safety net once everything (including iframes) has fully loaded.
  window.addEventListener("load", syncFrames);
})();
