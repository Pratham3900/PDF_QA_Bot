import React, { useState, useEffect } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import { Document, Page, pdfjs } from "react-pdf";
import "bootstrap/dist/css/bootstrap.min.css";
import {
  Container,
  Row,
  Col,
  Button,
  Form,
  Card,
  Spinner,
  Navbar,
} from "react-bootstrap";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();

const API_BASE = process.env.REACT_APP_API_URL || "";
const THEME_STORAGE_KEY = "pdf-qa-bot-theme";

function App() {
  const [file, setFile] = useState(null);
  const [pdfs, setPdfs] = useState([]); // { name, doc_id, url }
  const [selectedDocs, setSelectedDocs] = useState([]);
  const [selectedPdf, setSelectedPdf] = useState(null);

  const [question, setQuestion] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [comparisonResult, setComparisonResult] = useState(null);

  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);
  const [processingPdf, setProcessingPdf] = useState(false);
  const [summarizing, setSummarizing] = useState(false);
  const [comparing, setComparing] = useState(false);

  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);

  // -------------------------------
  // Theme (merged conflict safely)
  // -------------------------------
  const [darkMode, setDarkMode] = useState(() => {
    const savedTheme = localStorage.getItem(THEME_STORAGE_KEY);
    return savedTheme ? JSON.parse(savedTheme) : false;
  });

  useEffect(() => {
    localStorage.setItem(THEME_STORAGE_KEY, JSON.stringify(darkMode));
    document.body.classList.toggle("dark-mode", darkMode);
  }, [darkMode]);

  // -------------------------------
  // Upload PDF
  // -------------------------------
  const uploadPDF = async () => {
    if (!file) return;

    setUploading(true);
    setProcessingPdf(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post(`${API_BASE}/upload`, formData);
      const url = URL.createObjectURL(file);

      setPdfs((prev) => [
        ...prev,
        { name: file.name, doc_id: res.data?.doc_id, url },
      ]);

      setFile(null);
      alert("PDF uploaded!");
    } catch {
      alert("Upload failed.");
    }

    setUploading(false);
    setProcessingPdf(false);
  };

  // -------------------------------
  // Toggle document selection
  // -------------------------------
  const toggleDocSelection = (doc_id) => {
    setComparisonResult(null);
    setSelectedDocs((prev) =>
      prev.includes(doc_id)
        ? prev.filter((id) => id !== doc_id)
        : [...prev, doc_id]
    );
  };

  // -------------------------------
  // Ask Question
  // -------------------------------
  const askQuestion = async () => {
    if (!question.trim() || selectedDocs.length === 0) return;

    setChatHistory((prev) => [...prev, { role: "user", text: question }]);
    setAsking(true);

    try {
      const res = await axios.post(`${API_BASE}/ask`, {
        question,
        doc_ids: selectedDocs,
      });

      setChatHistory((prev) => [
        ...prev,
        { role: "bot", text: res.data.answer },
      ]);
    } catch {
      setChatHistory((prev) => [
        ...prev,
        { role: "bot", text: "Error getting answer." },
      ]);
    }

    setQuestion("");
    setAsking(false);
  };

  // -------------------------------
  // Summarize
  // -------------------------------
  const summarizePDF = async () => {
    if (selectedDocs.length === 0) return;

    setSummarizing(true);
    try {
      const res = await axios.post(`${API_BASE}/summarize`, {
        doc_ids: selectedDocs,
      });
      setChatHistory((prev) => [
        ...prev,
        { role: "bot", text: res.data.summary },
      ]);
    } catch {
      alert("Error summarizing.");
    }
    setSummarizing(false);
  };

  // -------------------------------
  // Compare
  // -------------------------------
  const compareDocuments = async () => {
    if (selectedDocs.length < 2) return;

    setComparing(true);
    try {
      const res = await axios.post(`${API_BASE}/compare`, {
        doc_ids: selectedDocs,
      });

      if (selectedDocs.length === 2) {
        setComparisonResult(res.data.comparison);
      } else {
        setChatHistory((prev) => [
          ...prev,
          { role: "user", text: "Compare selected documents." },
          { role: "bot", text: res.data.comparison },
        ]);
      }
    } catch {
      alert("Error comparing documents.");
    }
    setComparing(false);
  };

  const selectedPdfs = pdfs.filter((p) => selectedDocs.includes(p.doc_id));

  // -------------------------------
  // UI classes
  // -------------------------------
  const pageBg = darkMode ? "bg-dark text-light" : "bg-light text-dark";
  const cardClass = darkMode
    ? "text-white border-secondary shadow"
    : "bg-white text-dark border-0 shadow-sm";

  const inputClass = darkMode
    ? "text-white border-secondary placeholder-white"
    : "";

  return (
    <div className={pageBg} style={{ minHeight: "100vh" }}>
      <Navbar bg={darkMode ? "dark" : "primary"} variant="dark">
        <Container className="d-flex justify-content-between">
          <Navbar.Brand>PDF Q&A Bot</Navbar.Brand>
          <Button
            variant="outline-light"
            onClick={() => setDarkMode(!darkMode)}
          >
            Toggle Theme
          </Button>
        </Container>
      </Navbar>

      <Container className="mt-4">
        {/* Upload */}
        <Card className={`mb-4 ${cardClass}`}>
          <Card.Body>
            <Form>
              <Form.Control
                type="file"
                className={inputClass}
                onChange={(e) => setFile(e.target.files[0])}
              />
              <Button
                className="mt-2"
                onClick={uploadPDF}
                disabled={!file || uploading}
              >
                {uploading ? <Spinner size="sm" /> : "Upload"}
              </Button>
            </Form>
          </Card.Body>
        </Card>

        {/* Selection */}
        {pdfs.length > 0 && (
          <Card className={`mb-4 ${cardClass}`}>
            <Card.Body>
              <h5>Select Documents</h5>
              {pdfs.map((pdf) => (
                <Form.Check
                  key={pdf.doc_id}
                  type="checkbox"
                  label={pdf.name}
                  checked={selectedDocs.includes(pdf.doc_id)}
                  onChange={() => toggleDocSelection(pdf.doc_id)}
                />
              ))}
            </Card.Body>
          </Card>
        )}

        {/* Chat */}
        <Card className={cardClass}>
          <Card.Body>
            <div style={{ maxHeight: 300, overflowY: "auto", marginBottom: 16 }}>
              {chatHistory.map((msg, i) => (
                <div key={i} className="mb-2">
                  <strong>{msg.role === "user" ? "You" : "Bot"}:</strong>
                  <ReactMarkdown>{msg.text}</ReactMarkdown>
                </div>
              ))}
            </div>

            <Form
              className="d-flex gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                askQuestion();
              }}
            >
              <Form.Control
                type="text"
                placeholder="Ask a question..."
                value={question}
                className={inputClass}
                onChange={(e) => setQuestion(e.target.value)}
                disabled={asking}
              />
              <Button disabled={asking || !question.trim()}>
                {asking ? <Spinner size="sm" /> : "Ask"}
              </Button>
            </Form>

            <Button
              variant="warning"
              className="mt-2 me-2"
              onClick={summarizePDF}
            >
              Summarize
            </Button>

            <Button
              variant="info"
              className="mt-2"
              disabled={selectedDocs.length < 2}
              onClick={compareDocuments}
            >
              Compare
            </Button>
          </Card.Body>
        </Card>
      </Container>
    </div>
  );
}

export default App;