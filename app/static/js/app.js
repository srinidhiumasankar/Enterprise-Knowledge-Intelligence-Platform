/**
 * app.js
 * ------
 * Client-side JavaScript for the Enterprise Knowledge Intelligence Platform.
 *
 * Phase 2 scope:
 *   - Navbar background transition on scroll
 *   - Dashboard card staggered entrance animation (IntersectionObserver)
 *   - CTA button feedback (toast-style console notice — no business logic)
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
   Timezone & Datetime Utilities
   Parses naive UTC datetime string from backend and forces UTC parsing
   so the browser parses it as the user's local timezone.
   ============================================================ */
window.parseUTCDate = function (isoString) {
  if (isoString === null || isoString === undefined || isoString === "") {
    return new Date();
  }
  if (isoString instanceof Date) {
    return isoString;
  }
  if (typeof isoString === "number") {
    const d = new Date(isoString);
    return isNaN(d.getTime()) ? new Date() : d;
  }
  if (typeof isoString !== "string") {
    return new Date();
  }
  let dateStr = isoString;
  if (!dateStr.endsWith("Z") && !/[+-]\d{2}:\d{2}$/.test(dateStr)) {
    dateStr = dateStr + "Z";
  }
  const parsed = new Date(dateStr);
  return isNaN(parsed.getTime()) ? new Date() : parsed;
};

window.normalizeTimestamp = function (value) {
  if (value instanceof Date) {
    return value;
  }
  if (typeof value === "number") {
    const d = new Date(value);
    return isNaN(d.getTime()) ? new Date() : d;
  }
  if (typeof value === "string") {
    return window.parseUTCDate(value);
  }
  return new Date();
};

/* ============================================================
   UI Helpers: Toast Notification & Loading Spinner
   ============================================================ */
window.showToast = function (message, type = "info") {
  const container = document.querySelector(".toast-container");
  if (!container) return;

  const toastId = "toast_" + Date.now();
  const bgClass =
    type === "success"
      ? "bg-success text-white"
      : type === "danger"
      ? "bg-danger text-white"
      : type === "warning"
      ? "bg-warning text-dark"
      : "bg-primary text-white";

  const iconClass =
    type === "success"
      ? "bi-check-circle-fill"
      : type === "danger"
      ? "bi-exclamation-triangle-fill"
      : type === "warning"
      ? "bi-exclamation-circle-fill"
      : "bi-info-circle-fill";

  const toastHTML = `
    <div id="${toastId}" class="toast align-items-center ${bgClass} border-0 shadow-lg" role="alert" aria-live="assertive" aria-atomic="true" data-bs-delay="4000">
      <div class="d-flex">
        <div class="toast-body d-flex align-items-center gap-2">
          <i class="bi ${iconClass}"></i>
          <span>${message}</span>
        </div>
        <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
      </div>
    </div>
  `;
  container.insertAdjacentHTML("beforeend", toastHTML);
  const element = document.getElementById(toastId);

  if (window.bootstrap && window.bootstrap.Toast) {
    const bsToast = new window.bootstrap.Toast(element);
    bsToast.show();
    element.addEventListener("hidden.bs.toast", () => {
      element.remove();
    });
  } else {
    // Custom vanilla CSS fallback when bootstrap bundle is not loaded or is delayed
    element.style.display = "block";
    element.style.opacity = "0";
    element.style.transition = "opacity 0.35s ease";
    // Trigger layout calculation
    element.offsetHeight;
    element.style.opacity = "1";

    const closeBtn = element.querySelector("[data-bs-dismiss='toast']");
    const removeToast = () => {
      element.style.opacity = "0";
      setTimeout(() => element.remove(), 350);
    };

    if (closeBtn) {
      closeBtn.addEventListener("click", removeToast);
    }

    setTimeout(removeToast, 4000);
  }
};

window.showSpinner = function () {
  const overlay = document.getElementById("loadingOverlay");
  if (overlay) {
    overlay.classList.remove("d-none");
    overlay.classList.add("d-flex");
  }
};

window.hideSpinner = function () {
  const overlay = document.getElementById("loadingOverlay");
  if (overlay) {
    overlay.classList.add("d-none");
    overlay.classList.remove("d-flex");
  }
};

/* ============================================================
   Dynamic Navbar State Control & Sync Helpers
   ============================================================ */
function renderLoggedOutNavbar(navLinks) {
  navLinks.innerHTML = `
    <li class="nav-item">
      <a class="nav-link" href="/login" id="navLogin">Login</a>
    </li>
    <li class="nav-item ms-lg-2">
      <a class="btn btn-primary px-3 py-1 text-white mt-2 mt-lg-0" href="/register" id="navRegister">Sign Up</a>
    </li>
  `;
}

function renderLoggedInNavbar(navLinks, email) {
  navLinks.innerHTML = `
    <li class="nav-item">
      <a class="nav-link active" href="/" id="navHome">Dashboard</a>
    </li>
    <li class="nav-item">
      <a class="nav-link" href="/docs" id="navDocs" target="_blank" rel="noopener">
        API Docs
      </a>
    </li>
    <li class="nav-item dropdown ms-lg-2">
      <a class="nav-link dropdown-toggle btn btn-outline-primary d-flex align-items-center gap-2 py-1 px-3 text-start mt-2 mt-lg-0" href="#" id="userDropdown" role="button" data-bs-toggle="dropdown" aria-expanded="false">
        <i class="bi bi-person-circle"></i> Account
      </a>
      <ul class="dropdown-menu dropdown-menu-end border-0 shadow-lg mt-2 p-2" aria-labelledby="userDropdown">
        <li>
          <div class="px-3 py-2 text-muted small border-bottom mb-2 text-truncate" id="navUserEmail" style="max-width: 200px;">
            ${email}
          </div>
        </li>
        <li>
          <a class="dropdown-item text-danger d-flex align-items-center gap-2 rounded py-2" href="#" id="navLogout">
            <i class="bi bi-box-arrow-right"></i> Logout
          </a>
        </li>
      </ul>
    </li>
  `;

  // Wire up logout handler
  const logoutBtn = document.getElementById("navLogout");
  if (logoutBtn) {
    logoutBtn.addEventListener("click", (e) => {
      e.preventDefault();
      auth.logout();
    });
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  const navLinks = document.getElementById("navLinks");
  if (!navLinks) return;

  const authenticated = await auth.isLoggedIn();
  if (authenticated && auth.currentUser) {
    renderLoggedInNavbar(navLinks, auth.currentUser.email);
  } else {
    renderLoggedOutNavbar(navLinks);
  }
});
