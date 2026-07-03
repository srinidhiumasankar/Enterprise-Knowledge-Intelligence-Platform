/**
 * auth.js
 * -------
 * Authentication management utility.
 * Wraps token persistence and route protection rules.
 */

"use strict";

const auth = {
  // To cache the profile query promise and avoid concurrent duplicate requests
  _validationPromise: null,
  currentUser: null,

  saveToken(token) {
    if (token) {
      localStorage.setItem("access_token", token);
    }
  },

  getToken() {
    return localStorage.getItem("access_token");
  },

  removeToken() {
    localStorage.removeItem("access_token");
    this.currentUser = null;
  },

  async isLoggedIn() {
    const token = this.getToken();
    if (!token) {
      this.currentUser = null;
      return false;
    }

    if (this._validationPromise) {
      return this._validationPromise;
    }

    this._validationPromise = (async () => {
      try {
        const user = await api.get("/api/auth/me");
        if (user && user.email) {
          this.currentUser = user;
          return true;
        }
        this.removeToken();
        return false;
      } catch (err) {
        console.warn("[EKIP] Token validation check failed:", err.message);
        this.removeToken();
        return false;
      } finally {
        this._validationPromise = null;
      }
    })();

    return this._validationPromise;
  },

  async login(email, password) {
    try {
      const response = await api.post("/api/auth/login", { email, password });
      if (response && response.access_token) {
        this.saveToken(response.access_token);
        return response;
      }
      throw new Error("Invalid response format from server.");
    } catch (err) {
      throw err;
    }
  },

  async register(fullName, email, password) {
    try {
      const payload = {
        email: email,
        password: password,
        full_name: fullName || null,
      };
      return await api.post("/api/auth/register", payload);
    } catch (err) {
      throw err;
    }
  },

  async logout() {
    try {
      if (this.getToken()) {
        await api.post("/api/auth/logout", {});
      }
    } catch (err) {
      console.warn("Logout request failed on server, clearing client anyway:", err);
    } finally {
      this.removeToken();
      const navLinks = document.getElementById("navLinks");
      if (navLinks) {
        navLinks.innerHTML = "";
      }
      window.location.href = "/login";
    }
  },

  async protectRoute() {
    const valid = await this.isLoggedIn();
    if (!valid) {
      this.removeToken();
      window.location.href = "/login?redirect=" + encodeURIComponent(window.location.pathname);
    }
  },

  async redirectIfLoggedIn() {
    const valid = await this.isLoggedIn();
    if (valid) {
      window.location.href = "/";
    }
  },
};
