/**
 * navigation.js
 * -------------
 * Sets up dynamic active states, populates the profile topbar information,
 * and configures the logout hooks.
 */

"use strict";

(function initNavigation() {
  document.addEventListener("DOMContentLoaded", async () => {
    // --- 1. Dynamic Active Item State ---
    const path = window.location.pathname;
    const sidebarLinks = document.querySelectorAll(".sidebar-menu-list .nav-link");

    sidebarLinks.forEach((link) => {
      const href = link.getAttribute("href");
      if (href === path || (path === "/" && link.dataset.navTarget === "dashboard")) {
        link.classList.add("active");
      } else {
        link.classList.remove("active");
      }
    });

    // --- 2. Wire Up Logout Triggers ---
    const logoutBtnTopbar = document.getElementById("topbarLogout");
    const logoutBtnSidebar = document.getElementById("menu-logout");

    const handleLogout = (e) => {
      e.preventDefault();
      if (typeof auth !== "undefined" && typeof auth.logout === "function") {
        auth.logout();
      } else {
        console.warn("[EKIP] Auth library not loaded, clearing local storage manually.");
        localStorage.removeItem("access_token");
        window.location.href = "/login";
      }
    };

    if (logoutBtnTopbar) {
      logoutBtnTopbar.addEventListener("click", handleLogout);
    }
    if (logoutBtnSidebar) {
      logoutBtnSidebar.addEventListener("click", handleLogout);
    }

    // --- 3. Load Logged In Profile Details ---
    if (typeof auth !== "undefined" && typeof auth.isLoggedIn === "function") {
      try {
        const loggedIn = await auth.isLoggedIn();
        if (loggedIn && auth.currentUser) {
          const user = auth.currentUser;

          // Elements
          const nameEl = document.getElementById("topbarUserName");
          const emailEl = document.getElementById("topbarUserEmail");
          const initialsEl = document.getElementById("topbarUserInitials");

          // Display name and email
          const displayName = user.full_name || "Enterprise User";
          const displayEmail = user.email || "user@company.com";

          if (nameEl) nameEl.textContent = displayName;
          if (emailEl) emailEl.textContent = displayEmail;

          // Generate initials
          if (initialsEl) {
            let initials = "U";
            if (user.full_name) {
              const parts = user.full_name.trim().split(/\s+/);
              if (parts.length >= 2) {
                initials = (parts[0][0] + parts[1][0]).toUpperCase();
              } else if (parts.length === 1 && parts[0].length > 0) {
                initials = parts[0][0].toUpperCase();
              }
            } else if (user.email && user.email.length > 0) {
              initials = user.email[0].toUpperCase();
            }
            initialsEl.textContent = initials;
          }
        }
      } catch (err) {
        console.error("[EKIP] Failed to load topbar profile details:", err);
      }
    }

    // --- 4. Populate Navbar Workspace Selector Dropdown ---
    async function populateWorkspaceSelector() {
      const wsBtn = document.getElementById("activeWorkspaceName");
      const wsDropdownList = document.getElementById("wsSelectorDropdownList");
      if (!wsBtn || !wsDropdownList) return;

      try {
        const activeWs = await api.get("/api/workspaces/current");
        if (activeWs) {
          wsBtn.textContent = activeWs.name;
        }

        const allWs = await api.get("/api/workspaces");
        if (allWs && allWs.items) {
          const header = wsDropdownList.querySelector(".dropdown-header");
          wsDropdownList.innerHTML = "";
          if (header) wsDropdownList.appendChild(header);

          allWs.items.forEach(ws => {
            const li = document.createElement("li");
            const activeClass = ws.id === activeWs.id ? "active fw-bold text-primary" : "text-white-50";
            li.innerHTML = `
              <a class="dropdown-item d-flex align-items-center gap-2 rounded py-2 ${activeClass}" href="#">
                <i class="bi bi-briefcase"></i> ${ws.name}
              </a>
            `;
            li.querySelector("a").addEventListener("click", async (e) => {
              e.preventDefault();
              if (ws.id === activeWs.id) return;
              try {
                await api.post(`/api/workspaces/${ws.id}/switch`);
                window.location.reload();
              } catch (err) {
                console.error("Failed to switch workspace: ", err);
              }
            });
            wsDropdownList.appendChild(li);
          });
        }
      } catch (err) {
        console.error("Failed to populate workspace selector:", err);
      }
    }

    await populateWorkspaceSelector();
  });
})();
