import { useEffect, useRef, useState } from "react";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

const navigation = [
  { id: "overview", label: "Overview", icon: "grid" },
  { id: "analysis", label: "Leaf analysis", icon: "scan" },
  { id: "retrieval", label: "Retrieval results", icon: "layers" },
  { id: "monitoring", label: "IoT monitoring", icon: "activity" },
  { id: "recent-activity", label: "Recent activity", icon: "clock" },
];

function Icon({ name, size = 20 }) {
  const paths = {
    grid: (
      <>
        <rect x="3.5" y="3.5" width="7" height="7" rx="1.5" />
        <rect x="13.5" y="3.5" width="7" height="7" rx="1.5" />
        <rect x="3.5" y="13.5" width="7" height="7" rx="1.5" />
        <rect x="13.5" y="13.5" width="7" height="7" rx="1.5" />
      </>
    ),
    scan: (
      <>
        <path d="M7 3.5H5a1.5 1.5 0 0 0-1.5 1.5v2M17 3.5h2A1.5 1.5 0 0 1 20.5 5v2M20.5 17v2a1.5 1.5 0 0 1-1.5 1.5h-2M7 20.5H5A1.5 1.5 0 0 1 3.5 19v-2" />
        <path d="M12 5.5v13M8.5 9c.9-1 2-1.5 3.5-1.5s2.6.5 3.5 1.5M8.5 15c.9 1 2 1.5 3.5 1.5s2.6-.5 3.5-1.5" />
      </>
    ),
    layers: (
      <>
        <path d="m12 3.5 8.5 4.7-8.5 4.7-8.5-4.7L12 3.5Z" />
        <path d="m3.5 12 8.5 4.7 8.5-4.7M3.5 16.5l8.5 4.7 8.5-4.7" />
      </>
    ),
    activity: (
      <>
        <path d="M3 12h4l2.2-6 4.1 12 2.2-6H21" />
      </>
    ),
    clock: (
      <>
        <circle cx="12" cy="12" r="8.5" />
        <path d="M12 7v5l3.5 2" />
      </>
    ),
    leaf: (
      <>
        <path d="M20.5 3.5C11 3.5 5 6.7 5 13.2a5.3 5.3 0 0 0 5.3 5.3c6.5 0 10.2-6 10.2-15Z" />
        <path d="M3.5 21c3-5.3 6.5-8.7 12-12" />
      </>
    ),
    upload: (
      <>
        <path d="M12 15V4.5M8 8.5l4-4 4 4" />
        <path d="M5 14.5v4A1.5 1.5 0 0 0 6.5 20h11a1.5 1.5 0 0 0 1.5-1.5v-4" />
      </>
    ),
    image: (
      <>
        <rect x="3.5" y="4" width="17" height="16" rx="2" />
        <circle cx="9" cy="9.5" r="1.5" />
        <path d="m4 17 5-5 3.5 3 2.5-2 5 4" />
      </>
    ),
    flask: (
      <>
        <path d="M9 3.5h6M10 3.5v6L5 18a1.7 1.7 0 0 0 1.5 2.5h11A1.7 1.7 0 0 0 19 18l-5-8.5v-6" />
        <path d="M7.5 15h9" />
      </>
    ),
    thermometer: (
      <>
        <path d="M14 14.8V5a3 3 0 0 0-6 0v9.8a5 5 0 1 0 6 0Z" />
        <path d="M11 12v5" />
      </>
    ),
    droplets: (
      <>
        <path d="M12 3.5s-6 6.7-6 11a6 6 0 0 0 12 0c0-4.3-6-11-6-11Z" />
        <path d="M9.5 15.5a2.5 2.5 0 0 0 2.5 2.5" />
      </>
    ),
    arrow: (
      <>
        <path d="M5 12h14M13 6l6 6-6 6" />
      </>
    ),
    close: (
      <>
        <path d="m6 6 12 12M18 6 6 18" />
      </>
    ),
    check: (
      <>
        <path d="m5 12 4.5 4.5L19 7" />
      </>
    ),
  };

  return (
    <svg
      aria-hidden="true"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {paths[name]}
    </svg>
  );
}

function ReferenceThumbnail({ match, onOpen }) {
  const [hasError, setHasError] = useState(false);
  const imageUrl = `${API_BASE_URL}/api/v1/references/${encodeURIComponent(match.reference_id)}/image`;

  return (
    <button
      className="retrieval-thumbnail"
      type="button"
      onClick={() => onOpen(match)}
      disabled={hasError}
      aria-label={`Enlarge reference image ${match.display_filename}`}
      title={hasError ? "Reference image unavailable" : "Enlarge reference image"}
    >
      {hasError ? (
        <span className="thumbnail-fallback" aria-label="Reference image unavailable">
          <Icon name="image" size={20} />
        </span>
      ) : (
        <img
          src={imageUrl}
          alt=""
          loading="lazy"
          onError={() => setHasError(true)}
        />
      )}
    </button>
  );
}

function App() {
  const [activeSection, setActiveSection] = useState("overview");
  const [selectedImage, setSelectedImage] = useState(null);
  const [uploadError, setUploadError] = useState("");
  const [uploadResult, setUploadResult] = useState(null);
  const [classificationStatus, setClassificationStatus] = useState("idle");
  const [classificationResult, setClassificationResult] = useState(null);
  const [classificationError, setClassificationError] = useState("");
  const [retrievalStatus, setRetrievalStatus] = useState("idle");
  const [retrievalMatches, setRetrievalMatches] = useState([]);
  const [retrievalError, setRetrievalError] = useState("");
  const [selectedReference, setSelectedReference] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState("");
  const inputRef = useRef(null);

  useEffect(() => {
    if (!selectedImage) {
      setPreviewUrl("");
      return undefined;
    }

    const objectUrl = URL.createObjectURL(selectedImage);
    setPreviewUrl(objectUrl);
    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  }, [selectedImage]);

  useEffect(() => {
    if (!selectedReference) return undefined;

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    function handleKeyDown(event) {
      if (event.key === "Escape") setSelectedReference(null);
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [selectedReference]);

  function chooseImage(file) {
    if (isUploading) return;
    setUploadError("");
    setUploadResult(null);
    setClassificationStatus("idle");
    setClassificationResult(null);
    setClassificationError("");
    setRetrievalStatus("idle");
    setRetrievalMatches([]);
    setRetrievalError("");
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setSelectedImage(null);
      setUploadError("That file is not an image. Choose a JPG, PNG, or another image file.");
      return;
    }

    setSelectedImage(file);
  }

  function handleFileChange(event) {
    chooseImage(event.target.files?.[0]);
    event.target.value = "";
  }

  function removeImage() {
    if (isUploading) return;
    setSelectedImage(null);
    setUploadError("");
    setUploadResult(null);
    setClassificationStatus("idle");
    setClassificationResult(null);
    setClassificationError("");
    setRetrievalStatus("idle");
    setRetrievalMatches([]);
    setRetrievalError("");
    if (inputRef.current) inputRef.current.value = "";
  }

  function handleDrop(event) {
    event.preventDefault();
    setIsDragging(false);
    if (isUploading) return;
    chooseImage(event.dataTransfer.files?.[0]);
  }

  async function handleAnalyze() {
    if (!selectedImage || isUploading) return;

    setIsUploading(true);
    setUploadError("");
    setUploadResult(null);
    setClassificationStatus("idle");
    setClassificationResult(null);
    setClassificationError("");
    setRetrievalStatus("idle");
    setRetrievalMatches([]);
    setRetrievalError("");

    const formData = new FormData();
    formData.append("file", selectedImage);

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/images/upload`, {
        method: "POST",
        body: formData,
      });

      let responseBody;
      try {
        responseBody = await response.json();
      } catch {
        responseBody = null;
      }

      if (!response.ok) {
        const detail =
          typeof responseBody?.detail === "string" ? responseBody.detail : "";
        if (response.status === 413) {
          throw new Error("This image is larger than the 10 MB upload limit. Choose a smaller image.");
        }
        if (response.status === 415) {
          throw new Error("This image format is not supported. Choose a JPEG, PNG, or WebP image.");
        }
        if (response.status === 400) {
          throw new Error(detail || "The image is empty or could not be read. Choose another image.");
        }
        if (response.status >= 500) {
          throw new Error("The server could not save this image. Please try again.");
        }
        throw new Error(detail || `The upload failed (HTTP ${response.status}).`);
      }

      if (
        !responseBody?.upload_id ||
        !responseBody?.original_filename ||
        !responseBody?.detected_format ||
        !Number.isInteger(responseBody?.width) ||
        !Number.isInteger(responseBody?.height) ||
        responseBody?.status !== "uploaded"
      ) {
        throw new Error("The server returned an unexpected upload response. Please try again.");
      }

      setUploadResult(responseBody);
      setClassificationStatus("loading");
      setRetrievalStatus("loading");

      const uploadId = encodeURIComponent(responseBody.upload_id);
      const classificationRequest = (async () => {
        try {
          const classificationResponse = await fetch(
            `${API_BASE_URL}/api/v1/images/${uploadId}/classify`,
            { method: "POST" },
          );
          let classificationBody;
          try {
            classificationBody = await classificationResponse.json();
          } catch {
            classificationBody = null;
          }

          if (!classificationResponse.ok) {
            const detail =
              typeof classificationBody?.detail === "string"
                ? classificationBody.detail
                : "";
            throw new Error(
              detail ||
                `Classification is currently unavailable (HTTP ${classificationResponse.status}).`,
            );
          }

          if (
            classificationBody?.upload_id !== responseBody.upload_id ||
            classificationBody?.status !== "predicted" ||
            typeof classificationBody?.predicted_class !== "string" ||
            !["FN", "-N", "-P", "-K"].includes(
              classificationBody.predicted_class,
            ) ||
            typeof classificationBody?.class_name !== "string" ||
            !classificationBody.class_name.trim() ||
            typeof classificationBody?.note !== "string" ||
            !classificationBody.note.trim()
          ) {
            throw new Error(
              "The classification service returned an unexpected response.",
            );
          }

          setClassificationResult(classificationBody);
          setClassificationStatus("success");
        } catch (error) {
          setClassificationError(
            error instanceof TypeError
              ? `Classification is currently unavailable because the backend could not be reached at ${API_BASE_URL}.`
              : error instanceof Error
                ? `Classification is currently unavailable. ${error.message}`
                : "Classification is currently unavailable. Please try again later.",
          );
          setClassificationStatus("error");
        }
      })();

      const retrievalRequest = (async () => {
        try {
          const retrievalResponse = await fetch(
            `${API_BASE_URL}/api/v1/images/${uploadId}/retrieve`,
            { method: "POST" },
          );

          let retrievalBody;
          try {
            retrievalBody = await retrievalResponse.json();
          } catch {
            retrievalBody = null;
          }

          if (!retrievalResponse.ok) {
            const detail =
              typeof retrievalBody?.detail === "string" ? retrievalBody.detail : "";
            if (retrievalResponse.status === 503) {
              throw new Error(
                detail || "Image uploaded successfully, but retrieval is currently unavailable.",
              );
            }
            if (retrievalResponse.status === 404) {
              throw new Error("Image uploaded successfully, but the stored image could not be found for retrieval.");
            }
            throw new Error(
              detail || `Image uploaded successfully, but retrieval failed (HTTP ${retrievalResponse.status}).`,
            );
          }

          if (
            retrievalBody?.upload_id !== responseBody.upload_id ||
            retrievalBody?.status !== "retrieved" ||
            retrievalBody?.label_semantics !== "dataset_provided_label" ||
            retrievalBody?.similarity_semantics !== "cosine_similarity_not_probability" ||
            !Array.isArray(retrievalBody?.matches) ||
            retrievalBody.matches.length > 5 ||
            retrievalBody.matches.some(
              (match) =>
                typeof match?.reference_id !== "string" ||
                typeof match?.dataset_label !== "string" ||
                typeof match?.display_filename !== "string" ||
                typeof match?.similarity_score !== "number" ||
                !Number.isFinite(match.similarity_score) ||
                match.similarity_score < -1 ||
                match.similarity_score > 1,
            )
          ) {
            throw new Error("The retrieval service returned an unexpected response. Your image was uploaded successfully.");
          }

          setRetrievalMatches(retrievalBody.matches);
          setRetrievalStatus(retrievalBody.matches.length ? "success" : "empty");
        } catch (error) {
          setRetrievalError(
            error instanceof TypeError
              ? `The retrieval service could not be reached at ${API_BASE_URL}.`
              : error instanceof Error
                ? error.message
                : "Retrieval failed. Please try again.",
          );
          setRetrievalStatus("error");
        }
      })();

      await Promise.all([classificationRequest, retrievalRequest]);
    } catch (error) {
      setUploadError(
        error instanceof TypeError
          ? `Could not reach the backend at ${API_BASE_URL}. Check that FastAPI is running and that the frontend uses a permitted local origin (localhost:5173 or 127.0.0.1:5173), then try again.`
          : error instanceof Error
            ? error.message
            : "The upload failed unexpectedly. Please try again.",
      );
    } finally {
      setIsUploading(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="Main navigation">
        <a
          className="brand"
          href="#overview"
          onClick={() => setActiveSection("overview")}
          aria-label="LeafLab overview"
        >
          <span className="brand-mark">
            <Icon name="leaf" size={23} />
          </span>
          <span className="brand-name">
            <strong>LeafLab</strong>
            <span>HYDROPONICS RESEARCH</span>
          </span>
        </a>

        <div className="nav-caption">WORKSPACE</div>
        <nav className="navigation">
          {navigation.map((item) => (
            <a
              key={item.id}
              className={`nav-link ${activeSection === item.id ? "is-active" : ""}`}
              href={`#${item.id}`}
              aria-current={activeSection === item.id ? "location" : undefined}
              onClick={() => setActiveSection(item.id)}
            >
              <Icon name={item.icon} size={19} />
              <span>{item.label}</span>
              {item.id === "analysis" && <span className="nav-step">01</span>}
            </a>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="project-card">
            <span className="project-card-icon">
              <Icon name="leaf" size={17} />
            </span>
            <p>Research project</p>
            <span>B.Tech · Final year</span>
          </div>
          <div className="sidebar-footer">
            <span className="footer-dot" />
            <span>Development environment</span>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">
            <span>Workspace</span>
            <span className="breadcrumb-divider">/</span>
            <strong>Hydroponics CBIR</strong>
          </div>
          <div className="topbar-status">
            <span className="topbar-status-dot" />
            Research preview
          </div>
        </header>

        <div className="content-wrap">
          <section className="overview-section" id="overview">
            <div className="page-heading">
              <div>
                <p className="eyebrow">PLANT HEALTH RESEARCH · PROJECT OVERVIEW</p>
                <h1>Hydroponics CBIR</h1>
                <p className="page-description">
                  Content-Based Image Retrieval with Deep Feature Fusion for
                  Nutrient Deficiency Diagnosis in IoT-Enabled Hydroponic Systems.
                </p>
              </div>
              <div className="heading-badge">
                <span className="heading-badge-icon">
                  <Icon name="leaf" size={19} />
                </span>
                <span>
                  <strong>Project workspace</strong>
                  <small>Frontend prototype</small>
                </span>
              </div>
            </div>

            <div className="system-banner">
              <span className="system-icon">
                <Icon name="activity" size={19} />
              </span>
              <div className="system-copy">
                <strong>System overview</strong>
                <span>
                  Image classification and visual retrieval are available.
                  Live sensor integrations are not connected yet.
                </span>
              </div>
              <span className="prototype-tag">
                <span /> PROTOTYPE
              </span>
            </div>
          </section>

          <section className="section-block" id="analysis" aria-labelledby="analysis-title">
            <div className="section-heading">
              <div>
                <p className="section-kicker">01 · IMAGE INPUT</p>
                <h2 id="analysis-title">Leaf image analysis</h2>
                <p>Upload a leaf image for an experimental class prediction and visual retrieval.</p>
              </div>
              <span className="section-icon">
                <Icon name="scan" size={21} />
              </span>
            </div>

            <div className="analysis-card">
              <div
                className={`dropzone ${isDragging ? "is-dragging" : ""} ${selectedImage ? "has-image" : ""}`}
                onDragEnter={(event) => {
                  event.preventDefault();
                  if (!isUploading) setIsDragging(true);
                }}
                onDragOver={(event) => event.preventDefault()}
                onDragLeave={(event) => {
                  if (!event.currentTarget.contains(event.relatedTarget)) {
                    setIsDragging(false);
                  }
                }}
                onDrop={handleDrop}
              >
                {selectedImage ? (
                  <div className="image-preview">
                    {previewUrl && (
                      <img src={previewUrl} alt={`Preview of ${selectedImage.name}`} />
                    )}
                    <div className="image-details">
                      <span className="preview-file-icon">
                        <Icon name="image" size={18} />
                      </span>
                      <span className="preview-file-copy">
                        <strong title={selectedImage.name}>{selectedImage.name}</strong>
                        <small>{(selectedImage.size / 1024 / 1024).toFixed(2)} MB · Image ready</small>
                      </span>
                      <button
                        className="icon-button"
                        type="button"
                        onClick={removeImage}
                        disabled={isUploading}
                        aria-label="Remove selected image"
                        title="Remove image"
                      >
                        <Icon name="close" size={18} />
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="upload-prompt">
                    <span className="upload-icon">
                      <Icon name="upload" size={22} />
                    </span>
                    <strong>Drop a leaf image here</strong>
                    <span className="upload-subtitle">or browse files from your device</span>
                    <label className="browse-button" htmlFor="leaf-image">
                      <Icon name="image" size={17} />
                      Choose image
                    </label>
                    <span className="file-hint">Image files only · JPG, PNG, WEBP</span>
                  </div>
                )}
                <input
                  ref={inputRef}
                  id="leaf-image"
                  className="visually-hidden"
                  type="file"
                  accept="image/*"
                  aria-label="Choose a leaf image"
                  disabled={isUploading}
                  onChange={handleFileChange}
                />
              </div>

              {uploadError && (
                <p className="feedback-message error-message" role="alert">
                  <Icon name="close" size={17} />
                  {uploadError}
                </p>
              )}
              {isUploading && (
                <p className="feedback-message info-message" role="status" aria-live="polite">
                  <span className="loading-spinner" aria-hidden="true" />
                  {uploadResult
                    ? "Upload complete. Preparing classification and visual retrieval…"
                    : "Uploading and validating your image…"}
                </p>
              )}
              {uploadResult && (
                <div className="upload-result" role="status" aria-live="polite">
                  <div className="upload-result-heading">
                    <span className="upload-result-check">
                      <Icon name="check" size={17} />
                    </span>
                    <div>
                      <strong>Image uploaded and validated</strong>
                      <span>
                        Classification is an experimental model prediction;
                        retrieved images are visual matches, not diagnoses.
                      </span>
                    </div>
                  </div>
                  <dl className="upload-result-details">
                    <div>
                      <dt>Upload ID</dt>
                      <dd>{uploadResult.upload_id}</dd>
                    </div>
                    <div>
                      <dt>Filename</dt>
                      <dd title={uploadResult.original_filename}>
                        {uploadResult.original_filename}
                      </dd>
                    </div>
                    <div>
                      <dt>Format</dt>
                      <dd>{uploadResult.detected_format}</dd>
                    </div>
                    <div>
                      <dt>Dimensions</dt>
                      <dd>
                        {uploadResult.width} × {uploadResult.height}
                      </dd>
                    </div>
                    <div>
                      <dt>Status</dt>
                      <dd>{uploadResult.status}</dd>
                    </div>
                  </dl>
                </div>
              )}

              {uploadResult && (
                <section
                  className="classification-result"
                  aria-labelledby="classification-result-title"
                >
                  <div className="classification-heading">
                    <div>
                      <p className="section-kicker">MODEL PREDICTION</p>
                      <h3 id="classification-result-title">
                        Nutrient classification
                      </h3>
                    </div>
                    {classificationStatus === "loading" && (
                      <span className="classification-loading" role="status">
                        <span className="loading-spinner" aria-hidden="true" />
                        Classifying
                      </span>
                    )}
                  </div>

                  {classificationStatus === "loading" && (
                    <p className="classification-state" role="status">
                      The CPU model is analyzing the uploaded image.
                    </p>
                  )}
                  {classificationStatus === "error" && (
                    <p className="feedback-message error-message" role="alert">
                      <Icon name="close" size={17} />
                      {classificationError} Your image upload remains successful.
                    </p>
                  )}
                  {classificationStatus === "success" && (
                    <>
                      <div className="classification-prediction">
                        <strong>{classificationResult.class_name}</strong>
                        <span>
                          Class code:{" "}
                          <code>{classificationResult.predicted_class}</code>
                        </span>
                      </div>
                      <p className="classification-disclaimer">
                        {classificationResult.note}
                      </p>
                    </>
                  )}
                </section>
              )}

              <div className="analysis-actions">
                <span className="privacy-note">
                  <Icon name="check" size={15} />
                  Stored by local backend
                </span>
                <button
                  className="analyze-button"
                  type="button"
                  disabled={!selectedImage || isUploading}
                  aria-busy={isUploading}
                  onClick={handleAnalyze}
                >
                  {isUploading ? "Uploading…" : "Upload image"}
                  {!isUploading && <Icon name="arrow" size={17} />}
                </button>
              </div>
            </div>
          </section>

          <section className="section-block" id="retrieval" aria-labelledby="retrieval-title">
            <div className="section-heading">
              <div>
                <p className="section-kicker">02 · SIMILARITY SEARCH</p>
                <h2 id="retrieval-title">Retrieval results</h2>
                <p>Up to five visually similar indexed reference leaves.</p>
              </div>
              <span className="section-icon">
                <Icon name="layers" size={21} />
              </span>
            </div>
            <div className="retrieval-card">
              {retrievalStatus === "loading" && (
                <div className="retrieval-feedback" role="status" aria-live="polite">
                  <span className="loading-spinner" aria-hidden="true" />
                  <div>
                    <strong>Finding similar reference images</strong>
                    <span>Your upload succeeded. The CPU retrieval search is running.</span>
                  </div>
                </div>
              )}
              {retrievalStatus === "error" && (
                <div className="retrieval-feedback retrieval-error" role="alert">
                  <span className="retrieval-state-mark" aria-hidden="true">!</span>
                  <div>
                    <strong>Retrieval is unavailable</strong>
                    <span>{retrievalError} Your image upload remains successful.</span>
                  </div>
                </div>
              )}
              {retrievalStatus === "empty" && (
                <div className="retrieval-feedback" role="status">
                  <span className="retrieval-state-mark" aria-hidden="true">—</span>
                  <div>
                    <strong>No reference matches were returned</strong>
                    <span>Your image was uploaded successfully, but the retrieval index returned no results.</span>
                  </div>
                </div>
              )}
              {retrievalStatus === "success" && (
                <>
                  <div className="retrieval-success-heading" role="status">
                    <strong>{retrievalMatches.length} similar reference {retrievalMatches.length === 1 ? "image" : "images"}</strong>
                    <span>
                      Visually similar reference images, ranked by cosine
                      similarity. Scores are not diagnostic confidence.
                    </span>
                  </div>
                  <ol className="retrieval-list">
                    {retrievalMatches.map((match, index) => (
                      <li className="retrieval-match" key={match.reference_id}>
                        <ReferenceThumbnail
                          match={match}
                          onOpen={setSelectedReference}
                        />
                        <span className="retrieval-rank">{index + 1}</span>
                        <div className="retrieval-match-copy">
                          <strong title={match.display_filename}>{match.display_filename}</strong>
                          <span>Dataset label: <b>{match.dataset_label}</b></span>
                        </div>
                        <div className="retrieval-score">
                          <strong>{match.similarity_score.toFixed(4)}</strong>
                          <span>similarity</span>
                        </div>
                      </li>
                    ))}
                  </ol>
                </>
              )}
              {retrievalStatus === "idle" && (
                <div className="retrieval-feedback" role="status">
                  <span className="retrieval-state-mark" aria-hidden="true">
                    <Icon name="image" size={18} />
                  </span>
                  <div>
                    <strong>Results will appear here</strong>
                    <span>Upload a leaf image to search the saved reference index.</span>
                  </div>
                </div>
              )}
              <p className="retrieval-disclaimer">
                Labels are supplied by the dataset and are not independently verified diagnoses.
                Similarity scores are not probabilities; these results do not confirm a nutrient deficiency.
              </p>
            </div>
          </section>

          <section className="section-block" id="monitoring" aria-labelledby="monitoring-title">
            <div className="section-heading">
              <div>
                <p className="section-kicker">03 · GROW ENVIRONMENT</p>
                <h2 id="monitoring-title">IoT monitoring</h2>
                <p>Hydroponic reservoir parameters from connected sensors.</p>
              </div>
              <span className="section-icon">
                <Icon name="activity" size={21} />
              </span>
            </div>
            <div className="sensor-grid">
              <SensorCard
                icon="flask"
                label="pH level"
                unit="pH"
                description="Acidity / alkalinity"
                color="green"
              />
              <SensorCard
                icon="droplets"
                label="Electrical conductivity"
                unit="mS/cm"
                description="Nutrient solution strength"
                color="blue"
              />
              <SensorCard
                icon="thermometer"
                label="Water temperature"
                unit="°C"
                description="Reservoir temperature"
                color="amber"
              />
            </div>
            <p className="sensor-disclaimer">
              <span className="disclaimer-dot" />
              No sensor readings are being shown. Live data requires IoT
              integration; no simulated values are displayed.
            </p>
          </section>

          <section
            className="section-block activity-section"
            id="recent-activity"
            aria-labelledby="activity-title"
          >
            <div className="section-heading">
              <div>
                <p className="section-kicker">04 · WORKSPACE LOG</p>
                <h2 id="activity-title">Recent activity</h2>
                <p>Your image analysis sessions will be listed here.</p>
              </div>
              <span className="section-icon">
                <Icon name="clock" size={21} />
              </span>
            </div>
            <div className="activity-empty">
              <span className="activity-empty-icon">
                <Icon name="clock" size={20} />
              </span>
              <div>
                <strong>No activity yet</strong>
                <p>Completed analyses will appear here when the retrieval system is connected.</p>
              </div>
            </div>
          </section>

          <footer className="page-footer">
            <span>LeafLab <span className="footer-separator">·</span> Academic research prototype</span>
            <span>Content-based plant image retrieval</span>
          </footer>
        </div>
        {selectedReference && (
          <div
            className="image-lightbox"
            role="presentation"
            onClick={(event) => {
              if (event.target === event.currentTarget) {
                setSelectedReference(null);
              }
            }}
          >
            <section
              className="lightbox-dialog"
              role="dialog"
              aria-modal="true"
              aria-labelledby="lightbox-title"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="lightbox-header">
                <div>
                  <h2 id="lightbox-title">{selectedReference.display_filename}</h2>
                  <p>
                    Dataset label: <strong>{selectedReference.dataset_label}</strong>
                    <span aria-hidden="true"> · </span>
                    Similarity: {selectedReference.similarity_score.toFixed(4)}
                  </p>
                </div>
                <button
                  className="lightbox-close"
                  type="button"
                  onClick={() => setSelectedReference(null)}
                  aria-label="Close enlarged image"
                  autoFocus
                >
                  <Icon name="close" size={20} />
                </button>
              </div>
              <div className="lightbox-image-wrap">
                {selectedReference.imageUnavailable ? (
                  <div className="lightbox-fallback" role="status">
                    <Icon name="image" size={34} />
                    <span>This reference image could not be loaded.</span>
                  </div>
                ) : (
                  <img
                    src={`${API_BASE_URL}/api/v1/references/${encodeURIComponent(selectedReference.reference_id)}/image`}
                    alt={`Reference image ${selectedReference.display_filename}`}
                    onError={() =>
                      setSelectedReference((current) =>
                        current?.reference_id === selectedReference.reference_id
                          ? { ...current, imageUnavailable: true }
                          : current,
                      )
                    }
                  />
                )}
              </div>
              <p className="lightbox-disclaimer">
                Dataset labels are not confirmed diagnoses. Similarity scores are not probabilities.
              </p>
            </section>
          </div>
        )}
      </main>
    </div>
  );
}

function SensorCard({ icon, label, unit, description, color }) {
  return (
    <article className="sensor-card">
      <div className="sensor-card-top">
        <span className={`sensor-icon sensor-${color}`}>
          <Icon name={icon} size={19} />
        </span>
        <span className="simulated-badge">SIMULATED</span>
      </div>
      <h3>{label}</h3>
      <div className="sensor-reading">
        <span>—</span>
        <small>{unit}</small>
      </div>
      <p>{description}</p>
      <div className="sensor-card-footer">
        <span className="sensor-offline-dot" />
        No live sensor connected
      </div>
    </article>
  );
}

export default App;
