/**
 * sidebar.js
 * ----------
 * Orchestrates sidebar events, desktop expand/collapse persistence,
 * and mobile drawer toggle transitions.
 */

"use strict";

(function initSidebar() {
  document.addEventListener("DOMContentLoaded", () => {
    const appContainer = document.querySelector(".app-container");
    const btnSidebarCollapse = document.getElementById("btnSidebarCollapse");
    const btnSidebarToggleMobile = document.getElementById("btnSidebarToggleMobile");
    const btnSidebarCloseMobile = document.getElementById("btnSidebarCloseMobile");
    const sidebarBackdrop = document.getElementById("sidebarBackdrop");

    if (!appContainer) return;

    // --- Desktop: Collapse / Expand Persistence ---
    const STORAGE_KEY = "ekip_sidebar_collapsed";

    // Load persisted state
    const isCollapsed = localStorage.getItem(STORAGE_KEY) === "true";
    if (isCollapsed) {
      appContainer.classList.add("sidebar-collapsed");
    } else {
      appContainer.classList.remove("sidebar-collapsed");
    }

    if (btnSidebarCollapse) {
      btnSidebarCollapse.addEventListener("click", () => {
        appContainer.classList.toggle("sidebar-collapsed");
        const currentlyCollapsed = appContainer.classList.contains("sidebar-collapsed");
        localStorage.setItem(STORAGE_KEY, currentlyCollapsed);
      });
    }

    // --- Mobile: Off-canvas drawer ---
    const openMobileSidebar = () => {
      appContainer.classList.add("mobile-sidebar-open");
    };

    const closeMobileSidebar = () => {
      appContainer.classList.remove("mobile-sidebar-open");
    };

    if (btnSidebarToggleMobile) {
      btnSidebarToggleMobile.addEventListener("click", openMobileSidebar);
    }

    if (btnSidebarCloseMobile) {
      btnSidebarCloseMobile.addEventListener("click", closeMobileSidebar);
    }

    if (sidebarBackdrop) {
      sidebarBackdrop.addEventListener("click", closeMobileSidebar);
    }

    // Automatically close mobile sidebar on viewport width resize to desktop
    window.addEventListener("resize", () => {
      if (window.innerWidth >= 768) {
        closeMobileSidebar();
      }
    });
  });
})();
