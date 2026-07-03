/**
 * api.js
 * ------
 * Centralized API utility client for making HTTP requests.
 * Handles automatic JWT insertion, JSON parsing, timeout control, and 401 token expiry.
 */

"use strict";

const api = {
  /**
   * Internal request orchestrator.
   * @param {string} endpoint - The target URL path
   * @param {object} options - Fetch configurations (headers, method, body, timeout)
   */
  async request(endpoint, options = {}) {
    const token = localStorage.getItem("access_token");
    const headers = {
      "Content-Type": "application/json",
      ...options.headers,
    };

    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }

    const controller = new AbortController();
    const timeout = options.timeout || 15000;
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    const config = {
      ...options,
      headers,
      signal: controller.signal,
    };

    try {
      const response = await fetch(endpoint, config);
      clearTimeout(timeoutId);

      // Handle token expiration/unauthorized states globally
      if (response.status === 401) {
        localStorage.removeItem("access_token");
        const path = window.location.pathname;
        if (path !== "/login" && path !== "/register") {
          window.location.href = "/login?expired=true";
        }
      }

      if (!response.ok) {
        let errorMessage = `Request failed with status ${response.status}`;
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorMessage;
        } catch (_) {}
        throw new Error(errorMessage);
      }

      const text = await response.text();
      return text ? JSON.parse(text) : null;
    } catch (err) {
      clearTimeout(timeoutId);
      if (err.name === "AbortError") {
        throw new Error("Request timed out. Please try again.");
      }
      throw err;
    }
  },

  get(endpoint, options = {}) {
    return this.request(endpoint, { ...options, method: "GET" });
  },

  post(endpoint, body, options = {}) {
    return this.request(endpoint, {
      ...options,
      method: "POST",
      body: JSON.stringify(body),
    });
  },

  put(endpoint, body, options = {}) {
    return this.request(endpoint, {
      ...options,
      method: "PUT",
      body: JSON.stringify(body),
    });
  },

  delete(endpoint, options = {}) {
    return this.request(endpoint, { ...options, method: "DELETE" });
  },
};
