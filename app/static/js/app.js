/**
 * app.js
 * ------
 * Client-side JavaScript for the Enterprise Knowledge Intelligence Platform.
 *
 * Phase 2 scope:
 *   - Navbar background transition on scroll
 *   - Dashboard card staggered entrance animation (IntersectionObserver)
 *   - CTA button feedback (toast-style console notice — no business logic)
 *   - API health-check ping against GET /
 *
 * No external dependencies — vanilla ES6+ only.
 */

"use strict";

/* ============================================================
   Navbar: tighten background opacity when user scrolls down
   ============================================================ */
(function initNavbarScroll() {
  const navbar = document.getElementById("mainNav");
  if (!navbar) return;

  const onScroll = () => {
    if (window.scrollY > 24) {
      navbar.classList.add("scrolled");
    } else {
      navbar.classList.remove("scrolled");
    }
  };

  window.addEventListener("scroll", onScroll, { passive: true });

  // Run once on load to catch a page reload mid-scroll
  onScroll();
})();

/* ============================================================
   Dashboard Cards: staggered fade-up entrance on scroll.
   Uses IntersectionObserver — no layout shift.
   ============================================================ */
(function initCardAnimations() {
  const cards = document.querySelectorAll(".dashboard-card");
  if (!cards.length) return;

  // Hide cards before they enter the viewport
  cards.forEach((card) => {
    card.style.opacity = "0";
    card.style.transform = "translateY(20px)";
    card.style.transition = "opacity 0.55s cubic-bezier(0.16,1,0.3,1), transform 0.55s cubic-bezier(0.16,1,0.3,1)";
  });

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry, idx) => {
        if (entry.isIntersecting) {
          // Stagger each card by 90 ms
          const delay = idx * 90;
          setTimeout(() => {
            entry.target.style.opacity = "1";
            entry.target.style.transform = "translateY(0)";
          }, delay);
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.1, rootMargin: "0px 0px -40px 0px" }
  );

  cards.forEach((card) => observer.observe(card));
})();

/* ============================================================
   CTA Buttons: inform the user these features are coming soon.
   No navigation or business logic — Phase 2 is UI only.
   ============================================================ */
(function initCTAButtons() {
  const btnUpload = document.getElementById("btnUpload");
  const btnChat   = document.getElementById("btnChat");

  const notify = (message) => {
    // Console notice only — no alert(), no side-effects
    console.info("[EKIP]", message);
  };

  if (btnUpload) {
    btnUpload.addEventListener("click", () => {
      notify("Document upload will be available in a future phase.");
    });
  }

  if (btnChat) {
    btnChat.addEventListener("click", () => {
      notify("AI Chat interface will be available in a future phase.");
    });
  }
})();

/* ============================================================
   API Health Check — confirms FastAPI server is reachable.
   Logs result to the browser console only.
   ============================================================ */
(function checkAPIHealth() {
  fetch("/", { method: "HEAD" })
    .then((response) => {
      if (response.ok) {
        console.info("[EKIP] Server health check passed ✓ (HTTP", response.status + ")");
      } else {
        console.warn("[EKIP] Server responded with unexpected status:", response.status);
      }
    })
    .catch((err) => {
      console.error("[EKIP] Could not reach the server:", err.message);
    });
})();
