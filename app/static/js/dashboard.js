/**
 * dashboard.js
 * ------------
 * Handles global dashboard interactions and keyboard shortcut listeners.
 */

"use strict";

(function initDashboard() {
  document.addEventListener("DOMContentLoaded", () => {
    // --- 1. Keyboard Shortcut: Ctrl + / focuses topbar search ---
    document.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "/") {
        e.preventDefault();
        const searchInput = document.getElementById("global-search-input");
        if (searchInput) {
          searchInput.focus();
          searchInput.select();
        }
      }
    });

    // --- 2. Staggered card entrance overlay load ---
    const mainArea = document.getElementById("dashboardMainArea");
    if (mainArea) {
      mainArea.classList.add("animate-fade-in-up");
    }
  });
})();
