/**
 * dashboard.js
 * ------------
 * Handles global dashboard interactions and keyboard shortcut listeners.
 */

"use strict";

(function initDashboard() {
  document.addEventListener("DOMContentLoaded", () => {


    // --- 2. Staggered card entrance overlay load ---
    const mainArea = document.getElementById("dashboardMainArea");
    if (mainArea) {
      mainArea.classList.add("animate-fade-in-up");
    }
  });
})();
