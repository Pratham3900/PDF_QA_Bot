const express = require("express");
const cors = require("cors");
const multer = require("multer");
const axios = require("axios");
const axiosRetry = require("axios-retry").default;
const path = require("path");
require("dotenv").config();

// ------------------------------------------------------------------
// CONFIGURATION
// ------------------------------------------------------------------
const API_REQUEST_TIMEOUT = parseInt(
  process.env.API_REQUEST_TIMEOUT || "45000",
  10
);
const MAX_RETRY_ATTEMPTS = parseInt(
  process.env.MAX_RETRY_ATTEMPTS || "3",
  10
);

// ------------------------------------------------------------------
// APP SETUP
// ------------------------------------------------------------------
const app = express();
app.set("trust proxy", 1);
app.use(cors());
app.use(express.json());

// ------------------------------------------------------------------
// AXIOS RETRY CONFIG
// ------------------------------------------------------------------
axiosRetry(axios, {
  retries: MAX_RETRY_ATTEMPTS,
  retryDelay: axiosRetry.exponentialDelay,
  retryCondition: (error) => {
    return (
      axiosRetry.isNetworkOrIdempotentRequestError(error) ||
      error.code === "ECONNABORTED" ||
      (error.response && error.response.status >= 500)
    );
  },
  onRetry: (retryCount, error, requestConfig) => {
    console.warn(
      `Retry ${retryCount} for ${requestConfig.url} - ${error.message}`
    );
  },
});

// ------------------------------------------------------------------
// MULTER CONFIG (PDF UPLOADS)
// ------------------------------------------------------------------
const upload = multer({ dest: "uploads/" });

// ------------------------------------------------------------------
// ROUTE: UPLOAD PDF
// ------------------------------------------------------------------
app.post("/upload", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res
        .status(400)
        .json({ error: "No file uploaded. Use form field name 'file'." });
    }

    const filePath = path.join(__dirname, req.file.path);

    await axios.post(
      "http://localhost:5000/process-pdf",
      { filePath },
      { timeout: API_REQUEST_TIMEOUT }
    );

    res.json({ message: "PDF uploaded & processed successfully!" });
  } catch (err) {
    const details = err.response?.data || err.message;
    console.error("Upload processing failed:", details);

    if (err.code === "ECONNABORTED" || err.response?.status === 504) {
      return res.status(504).json({
        error: "Request timed out",
        details:
          "PDF processing took too long. Try a smaller PDF or try again.",
      });
    }

    res.status(500).json({
      error: "PDF processing failed",
      details,
    });
  }
});

// ------------------------------------------------------------------
// ROUTE: ASK QUESTION (WITH VALIDATION)
// ------------------------------------------------------------------
app.post("/ask", async (req, res) => {
  const { question } = req.body;

  // ---- Input Validation ----
  if (!question || typeof question !== "string") {
    return res
      .status(400)
      .json({ error: "Question is required and must be a string" });
  }

  if (!question.trim()) {
    return res.status(400).json({ error: "Question cannot be empty" });
  }

  if (question.length > 2000) {
    return res
      .status(400)
      .json({ error: "Question too long (max 2000 characters)" });
  }

  try {
    const startTime = Date.now();

    const response = await axios.post(
      "http://localhost:5000/ask",
      { question: question.trim() },
      { timeout: API_REQUEST_TIMEOUT }
    );

    console.log(
      `Question answered in ${Date.now() - startTime}ms`
    );

    res.json({ answer: response.data.answer });
  } catch (err) {
    console.error("Ask failed:", err.message);

    if (err.code === "ECONNABORTED" || err.response?.status === 504) {
      return res.status(504).json({
        error: "Request timed out",
        details:
          "The question took too long to process. Try a simpler question.",
      });
    }

    res.status(err.response?.status || 500).json({
      error: "Error answering question",
      details: err.response?.data || err.message,
    });
  }
});

// ------------------------------------------------------------------
// ROUTE: SUMMARIZE PDF
// ------------------------------------------------------------------
app.post("/summarize", async (req, res) => {
  try {
    const startTime = Date.now();

    const response = await axios.post(
      "http://localhost:5000/summarize",
      req.body || {},
      { timeout: API_REQUEST_TIMEOUT }
    );

    console.log(
      `Summarization completed in ${Date.now() - startTime}ms`
    );

    res.json({ summary: response.data.summary });
  } catch (err) {
    const details = err.response?.data || err.message;
    console.error("Summarization failed:", details);

    if (err.code === "ECONNABORTED" || err.response?.status === 504) {
      return res.status(504).json({
        error: "Request timed out",
        details: "Summarization took too long. Please try again.",
      });
    }

    res.status(err.response?.status || 500).json({
      error: "Error summarizing PDF",
      details,
    });
  }
});

// ------------------------------------------------------------------
// START SERVER
// ------------------------------------------------------------------
app.listen(4000, () => {
  console.log("Backend running on http://localhost:4000");
});