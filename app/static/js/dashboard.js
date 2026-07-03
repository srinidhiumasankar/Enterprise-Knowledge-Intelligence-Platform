/**
 * dashboard.js
 * ------------
 * Handles global dashboard interactions, keyboard shortcut listeners (e.g. Ctrl + / for search focus),
 * and simple interactive events for components.
 */

"use strict";

(function initDashboard() {
  document.addEventListener("DOMContentLoaded", () => {
    // --- 1. Keyboard Shortcut: Ctrl + / ---
    document.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "/") {
        e.preventDefault();
        if (typeof window.showToast === "function") {
          window.showToast("Semantic Search input query is locked in this phase.", "info");
        }
      }
    });

    // --- 2. Inform user on search click ---
    const searchContainer = document.querySelector(".search-container");
    if (searchContainer) {
      searchContainer.addEventListener("click", () => {
        if (typeof window.showToast === "function") {
          window.showToast("Semantic Search will be enabled in a future phase.", "info");
        }
      });
    }

    // --- 3. Staggered card entrance overlay load ---
    const mainArea = document.getElementById("dashboardMainArea");
    if (mainArea) {
      mainArea.classList.add("animate-fade-in-up");
    }
  });
})();
