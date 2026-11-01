// auth.js – client‑side helpers for login / signup and token handling

// Simple wrapper to get the stored JWT (if any)
function getToken() {
  return localStorage.getItem('evorag_token');
}

function setToken(token) {
  localStorage.setItem('evorag_token', token);
}

function clearToken() {
  localStorage.removeItem('evorag_token');
}

// Generic POST helper that adds Authorization header when token exists
async function postJSON(url, data, auth = false) {
  const headers = { 'Content-Type': 'application/json' };
  if (auth) {
    const token = getToken();
    if (token) headers['Authorization'] = `Bearer ${token}`;
  }
  const resp = await fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(data),
  });
  const json = await resp.json();
  if (!resp.ok) throw new Error(json.detail || JSON.stringify(json));
  return json;
}

// Login form handling
const loginForm = document.getElementById('login-form');
if (loginForm) {
  loginForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const email = loginForm.email.value.trim();
    const password = loginForm.password.value;
    const errorDiv = document.getElementById('login-error');
    try {
      const tokenResp = await postJSON('/auth/login', { username: email, password });
      setToken(tokenResp.access_token);
      // After login, redirect to onboarding if not completed, else to chat (placeholder)
      const prefResp = await fetch('/preferences/', {
        headers: { Authorization: `Bearer ${getToken()}` },
      });
      const pref = await prefResp.json();
      if (pref.onboarding_completed) {
        window.location.href = '/static/chat.html';
      } else {
        window.location.href = '/static/onboarding.html';
      }
    } catch (err) {
      errorDiv.textContent = err.message;
      errorDiv.style.display = 'block';
    }
  });
}

// Signup form handling
const signupForm = document.getElementById('signup-form');
if (signupForm) {
  signupForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const fullName = signupForm.full_name.value.trim();
    const email = signupForm.email.value.trim();
    const password = signupForm.password.value;
    const confirm = signupForm.confirm_password.value;
    const errorDiv = document.getElementById('signup-error');
    if (password !== confirm) {
      errorDiv.textContent = 'Passwords do not match.';
      errorDiv.style.display = 'block';
      return;
    }
    try {
      await postJSON('/auth/signup', { email, full_name: fullName, password });
      // Auto‑login after successful signup
      const tokenResp = await postJSON('/auth/login', { username: email, password });
      setToken(tokenResp.access_token);
      window.location.href = '/static/onboarding.html';
    } catch (err) {
      errorDiv.textContent = err.message;
      errorDiv.style.display = 'block';
    }
  });
}

// Expose token helpers for other pages
window.EvoRAG = { getToken, setToken, clearToken };

