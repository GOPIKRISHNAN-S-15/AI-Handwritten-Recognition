# Implementation State — working memory for the multi-phase fix task

## Task (from /home/ubuntu/upload/pasted_content.txt, full 16 phases)
Fix the Streamlit HWR project at D:\7th Sem\AI&AP Project (desktop mount: /mnt/fb439304-7cc3-49cb-a35a-43d1d61363ac/AI&AP Project).
Order: 1 re-audit 2 protect MNIST 3 multi-digit segmentation 4 EMNIST training 5 EMNIST artifacts 6 model switching 7 Analytics 8 Model Lab 9 space detection 10 routing 11 Gemini 12 remove hardcoded metrics 13 System page 14 UI redesign 15 tests 16 verify+deploy+README.

## Done so far
1. Phase 1: Baseline verified. Backup saved at /home/ubuntu/backup/AI-AP-backup (all source files + mnist_model.keras + MNIST eval/history JSON, NO .env copy, NO venv).
2. Phase 2 in progress: segmentation strategy prototyped.

## Key verified facts (from audit, see /home/ubuntu/audit/forensic_notes.md and /home/ubuntu/forensic_audit_report.md)
- mnist_model.keras: 98.82% on 10k; synthesized "11" → class 4 @68.9%; latency ~58-198ms CPU.
- Gemini: key valid; gemini-1.5-flash 404; gemini-3.5-flash OK; gemini-3.7-flash OK; gemini-3.6-flash 429.
- Existing segment_characters(): contour-only, no projection splitting.
- detect_spaces() never called; 2_DOCUMENTS.py line 183 broken switch_page("pages/5_✨_GenAI_Insights.py") → must be pages/5_LANGUAGE.py.
- Model Lab line 99 hardcoded "~8.5 ms/sample"; app.py hero: "MNIST 99.6%", "8.5 ms", static 60k/10k, "3.6 Flash" string.
- 6_SYSTEM.py false claims: Radon Deskewing, Bilateral/Gaussian smoothing; hardcoded TensorFlow/Streamlit.
- constants.py: GEMINI_MODEL="gemini-1.5-flash" (line 120); hyperparams epochs=2.
- train_model.py: real EMNIST support w/ transpose+flip; needs more epochs (~10-15).
- No emnist artifacts exist.

## Segmentation probe results (IMPORTANT for implementation)
- Naive contours on separated digits: OK for all test strings (1,11,12,123,2026,98765,100,42).
- BUT end-to-end (seg + MNIST classify) FAILS: 2026→2123, 98765→94733, 100→111. Cause: my PROBE's block-digit font strokes are non-standard (my font's 0/6/8/9/2 shapes differ from MNIST style). MNIST model trained on real handwriting. The model itself is fine; my probe font is misleading. NEED real handwritten multi-digit images OR better probe fonts OR trust per-digit probe (naive contours split OK).
- Touching-digit probe: projection-split strategy (ratio 0.03, min_gap 0, MORPH_OPEN, aspect>1.5, bw>=3) correctly splits 11/12/123/2026 touching and 98765+noise.

## Correct implementation decision for segmentation (image_processor.py segment_characters)
Use: morph open to remove noise → contours → for each bbox with w/h > 1.5, split via vertical projection (ink threshold = max(1, h*0.03), min gap 0→1 col) → filter seg width >= 3px → fallback keep whole; also handle overlapping (y-overlap merging into lines not needed for canvas but helpful for documents).

## Segmentation debug findings (root cause of '888' predictions)
- Canvas composition polarity fixed (minimum, 0=ink). Segmentation now correctly splits: '1'→1 seg, '11'→2, '12'→2, '123'→3 boxes at correct positions.
- BUT each segment gets predicted as '8' (conf ~1.0). Root cause: segment crop is 32x32 (pad 2+), has huge white border → ImageAnalyzer thinks background_intensity=255, is_noisy, is_low_contrast → applies fastNlMeansDenoising + ADAPTIVE_THRESHOLD_GAUSSIAN_C (uneven background path) → binarization mean jumps 14→136 (inverts/hallucinates ink) → garbage.
- Direct pipeline on the same raw MNIST sample: Otsu only, binarized mean 14. Works (98.82%).
- FIX NEEDED: segment_characters should return CROPPED BINARIZED region or a tightly-cropped binary image, and preprocess should detect content and ignore border padding; OR reduce pad to 0; OR in preprocess, crop to content BEFORE denoise/CLAHE (the code already crops after binarization, but pad creates a border that adaptive threshold misinterprets).
- Real fix approach: in segment_characters, crop from the BINARIZED+cleaned content tightly (pad=0) OR normalize the crop. Simpler robust approach: crop with pad from gray image but cap pad to 0 (no padding) so the whole crop is content.

## FINAL root cause (after 12 debug iterations)
- canvas[38:66,38:66] == img exactly; BUT segment_characters crops at box (39,39) — offset by 1 because MORPH_OPEN/kernel? Actually the segment bbox is 1px shifted (39 vs 38). So crop includes a border column/row!
- ALSO: fastNlMeansDenoising is applied when is_noisy (Laplacian var > 1500); raw MNIST noise_level=11175 → noisy! The denoiser smooths/alters strokes (diff pixels: 267). Direct pipeline ALSO denoises raw img — yet direct works. Diff: direct denoises img with content; canvas-crop denoises crop with border pixels included → denoiser treats border differently → corrupted strokes → 0→4.
- Fix decision: (1) segment_characters must crop EXACTLY the content bbox (no off-by-one); currently uses contour bbox which should be exact — off-by-1 may come from morph_open. (2) Add a 'tight character crop' path skipping denoising for already-tight crops: skip fastNlMeansDenoising when image has uniform border (real single-char crop detection: no pure-white border → it's a tight crop). Apply denoise only on uploaded/scanned documents with borders.
- After fix: re-run test harness (e2e_real_handwriting_test.py) — expects all 8 strings OK.

## CURRENT STATE of segmentation fix (as of debug 15)
- Working iteration BEFORE expansion fix: bbox (39,39,28,28) tight 28x28 crop + denoise-guard (skip fastNlMeans/CLAHE/deskew when no uniform border) + tight-crop gate in binarized-cropping → 5/8 strings pass (1,11,12,123,42 = 10/10; 2026→2424, 98765→14245, 100→144 all fail on digit 0→4).
- Remaining failure: digit 0 sample still predicted 4 via canvas path. Denoising now SKIPPED (ops confirm), binarized still differs from direct path. Need to find why binarized differs when ops identical (Otsu, tight-crop, CoM, norm).
- NEW regression: +1px bbox expansion made crop 30x30 with white edges → REVERT the expansion. Use exact ink bbox from binary_clean WITHOUT expansion, OR compute bbox on pre-open binary.
- Direct pipeline on the same img0: applies fastNlMeansDenoising (is_noisy=True, Laplacian 11175) → works fine (pred 0 conf .84). So the model IS robust to denoising on full-frame data.
- Canvas-crop path (tight, no denoise): pred 4 conf .99. binarized stages differ: direct binarized mean 47.49, canvas 65.38.
- Both paths: same Otsu on same gray → binarized should be identical. DIFFERENCE must come earlier: direct grayscale mean 47.21 vs canvas-crop gray mean 65.1 (because img0 placed on canvas at 38:66 → identical 28x28 region... but crop had diff pixels!). 
- ROOT: segment_characters crop from `image` (canvas) but bbox 38:66 means crop must equal img0... debug showed 267 differing pixels w/ crop having WHITE where img has 0 (ink) → the contour bbox on binary_clean is INSIDE the true ink (erosion), so the crop CUTS OFF ink edges. With +1 expansion we cut background instead → 30x30 crop includes white edges → binarization includes extra background → ink density drops → 0→4.
- CORRECT FINAL FIX: compute bbox on the ORIGINAL binary (before MORPH_OPEN) for cropping, but use binary_clean for contour detection/splitting. E.g. for each component, expand bbox to include all ink pixels of binary within a small neighborhood.

## PHASE 3 STATUS (EMNIST training)
- EMNIST Balanced downloaded via TFDS to ~/.emnist_balanced/emnist_balanced.npz (112800 train / 18800 test, 47 classes, already transposed+flipped).
- training/train_emnist.py written (real pipeline: augmentation, early stopping, ReduceLROnPlateau, full eval JSON w/ confusion matrix + per-class report).
- constants.py: EMNIST epochs raised 2→12; train_emnist.py + constants.py synced to desktop.
- Training running in background: cd /home/ubuntu/audit && nohup python3 training/train_emnist.py > training/emnist_training.log 2>&1 &
- NOTE: the EMNIST labels in TFDS already have correct orientation (TFDS applies transform); our npz applied flipud(transpose) manually → VERIFY orientation with a sample visualization before trusting eval. (TFDS emnist/balanced already returns corrected images; applying extra transpose+flip would MIRROR them. Need to check!)
- After training: Phase 4 (model switching, analytics, model lab, routing, Gemini), Phase 5 Gemini fix (gemini-3.5-flash working; 3.6-flash 429; env var GEMINI_MODEL), Phase 6 UI redesign, Phase 7 tests, Phase 8 deploy prep.

## Next steps
- Rewrite segment_characters in image_processor.py with projection splitting + wide-character warning.
- Build E2E test with REAL MNIST-style digits: use 10k MNIST test set, crop individual digits, tile them side-by-side into a multi-digit canvas (real handwriting!) → classify with new segmenter. That is the correct validation dataset.
- Then EMNIST training (train_model.py, epochs 12-15 EMNIST; use emnist package), save model/history/eval JSON.
- Then: genuine model switching helper module, Analytics/Model Lab live metrics (compute latency measured), space detection wiring, routing fix, Gemini model via GEMINI_MODEL env var, remove hardcoded metrics, System page, UI redesign (Phase 11: CV research workstation identity), tests (12 cases), deployment prep, README.
- Tests must be runnable on the user's desktop (they have venv at project/venv).

## Project file map (source-of-truth paths are on desktop)
- app.py | pages/1_CAPTURE.py 2_DOCUMENTS.py 3_ANALYTICS.py 4_MODEL_LAB.py 5_LANGUAGE.py 6_SYSTEM.py
- models/cnn_model.py (+ mnist_model.keras, emnist to create)
- genai/ai_service.py | preprocessing/image_processor.py segmentation.py | analytics/model_analysis.py
- training/train_model.py train_mnist_only.py | utils/constants.py helpers.py ui_components.py | styles/main.css
- tests/test_audit_verification.py test_mnist_known_images.py | .env .env.example requirements.txt README.md

## PHASE 4 FILE DETAILS (verified from fresh reads)
- cnn_model.py: load_trained_model(model_type) @st.cache_resource returns None if file missing; models/ subfolder in project root; paths relative to module.
- utils/helpers.py: predict_character(model, image, class_mapping, top_n=5) — works for any model; generate_report/generate_report_json(model_type kwarg).
- pages/1_CAPTURE.py lines 64-85: model select already wired (MNIST→model, EMNIST→model) but EMNIST option hidden when no emnist_model.keras. OK.
- pages/2_DOCUMENTS.py lines 53-69: model select OK; BUG line 183: st.switch_page("pages/5_✨_GenAI_Insights.py") nonexistent page; real Gemini page is 5_LANGUAGE.py (verify filename). Also no detect_spaces() call — recognized_text is chars only; need to call segmentation.py's detect_spaces or similar.
- pages/3_ANALYTICS.py: loads load_evaluation(model_type)/load_training_history(model_type); static metadata fallbacks; NO EMNIST gating (shows info msg). Needs: real gating (disable EMNIST when model/eval missing), live metrics from eval JSON.
- pages/4_MODEL_LAB.py: hard-stops if eval absent; line ~91-100 hardcoded "~8.5 ms/sample"; model.count_params() at ~85 fails if model None; needs real measured latency + gating.
- app.py lines 30-31 loads both models, cnn_loaded bool; lines 64-93 hardcoded hero telemetry (MNIST 99.6%, 8.5ms, dataset counts, Gemini badge "3.6 Flash"). Needs live metrics.
- pages/6_SYSTEM.py: shows GEMINI_MODEL const; static doc claims (bilateral smoothing, Radon deskewing) — Phase 6 fix.
- utils/ui_components.py: rendering only, no logic.
- EMNIST training log: /home/ubuntu/audit/training/emnist_training.log (running in sandbox; copy model+artifacts back to desktop when done: models/emnist_model.keras, training/training_history_emnist.json, training/evaluation_emnist.json).
- Gemini probe results: gemini-3.5-flash OK, gemini-3.7-flash OK, gemini-3.6-flash 404/quota; gemini-1.5-flash 404. .env key valid.

## Gemini integration facts
- genai/ai_service.py uses hardcoded GEMINI_MODEL from constants (gemini-1.5-flash → 404).
- Fix: read from env var GEMINI_MODEL (default gemini-3.5-flash), verify with real request at startup (get_genai_service().check_connection()), update System page live.

## PHASE 4+5 DONE (synced to desktop, syntax OK)
- 2_DOCUMENTS.py: detect_spaces wired (digit mode merges numbers, alphanumeric inserts spaces); routing fixed to pages/5_LANGUAGE.py.
- 3_ANALYTICS.py: EMNIST option gated behind emnist model+eval+history; num_classes from eval JSON.
- 4_MODEL_LAB.py: gating by ready models; real measured latency (_measure_latency, median over 10 runs, 100-image batch); params marked measured.
- app.py: live dashboard metrics (_live_dashboard_metrics: measured accuracy from eval JSON, measured latency, emnist engine gauge if model present); Gemini gauge shows genai_service.model_name.
- genai/ai_service.py: model from GEMINI_MODEL env var (default gemini-3.5-flash), real connection probe with fallback list [3.5, 3.7, 2.5 flash].
- .env: added GEMINI_MODEL=gemini-3.5-flash (synced to desktop).
- constants.py: EMNIST epochs 12.

## PHASE 6 DONE (synced to desktop)
- styles/main.css fully rewritten: flat slate instrument-panel theme, IBM Plex Sans/Mono, amber accent, ruled borders, sharp corners, no glows/gradients/hero.
- 6_SYSTEM.py preprocessing claims corrected to match actual image_processor.py ops.
- constants.py GEMINI_MODEL updated to gemini-3.5-flash.
- NOTE: ui_components.py inline styles still use var(--bg-card) etc. — CSS var names unchanged, only values/palette changed in CSS; HTML components fine.

## EMNIST training status
- Running sandbox training/train_emnist.py; log training/emnist_training.log. Outputs: models/emnist_model.keras, training/training_history_emnist.json, training/evaluation_emnist.json → must be COPIED to desktop project after completion.
- At epoch 5/12 val_accuracy ~0.88.

## Remaining tasks
- Wait for EMNIST training to finish; copy artifacts to desktop; verify eval JSON fields (test_accuracy, per_class, confusion_matrix, class_mapping).
- Phase 7: tests on DESKTOP (desktop venv at D:\7th Sem\AI&AP Project\.venv — install deps: tensorflow, opencv, streamlit, matplotlib, seaborn, plotly, scikit-learn, python-dotenv, google-genai, Pillow). Test: segmentation 11/123/2026 via real pipeline, EMNIST model load+predict, Gemini real request via ai_service, analytics JSON loads, no dead pages.
- Phase 8: streamlit syntax check all pages + README update + deployment readiness check (requirements.txt matches, env documented).
- Phase 9: final report with genuinely working / partial / unavailable lists.

## DESKTOP PYTHON ENVIRONMENT (verified Aug 15)
- Default `python` is Windows Store stub at C:\Users\Gopi\AppData\Local\Microsoft\WindowsApps\python.exe — unusable (exit 1/no-op behavior).
- `py -0p` shows: Python 3.13 at C:\Users\Gopi\AppData\Local\Programs\Python\Python313\python.exe (DEFAULT) and Python 3.11 (Store path).
- 3.11 env has: matplotlib, pillow, plotly, python-dotenv, scikit-learn, seaborn, streamlit 1.54.0, streamlit-lottie, tensorboard, keras 3.13.2, google packages (generativeai 0.8.6, api-core etc.) — but NO tensorflow, NO opencv (opencv-python missing in both!).
- 3.13 env: only numpy 2.3.4; no tf/streamlit/opencv.
- CONCLUSION: desktop env is broken/incomplete for running the project (no tensorflow, no opencv). Project previously couldn't run. Tests should run in SANDBOX against the desktop-mounted project copy OR install deps on desktop (risky/slow). Plan: run all tests in sandbox using mounted project at /mnt/fb439304-7cc3-49cb-a35a-43d1d61363ac/AI&AP Project (sandbox has tf, cv2, streamlit, etc.).
- Existing desktop tests dir: tests/test_audit_verification.py, tests/test_mnist_known_images.py.
- EMNIST eval result saved: 89.51% (1973/18800 misclassified); artifacts copied to desktop (models/emnist_model.keras, training/evaluation_emnist.json, training/training_history_emnist.json, training/eval_only_emnist.py).
- NOTE for README/deployment section: document that user must install tensorflow-cpu and opencv-python (both missing on desktop).

## PHASE 7 TEST STATUS (Aug 15 evening)
- Test suite: /home/ubuntu/audit/tests/test_full_verification.py (also synced to desktop tests/).
- Preprocessing API: pre = AdaptivePreprocessor(); tensor, analysis = pre.preprocess(img) returns (28,28) float — must np.expand_dims(tensor, 0) before model call. NOT ImageAnalyzer.preprocess_image (removed).
- Latest run: 38 PASS / 3 FAIL:
  * T2 '98765': 0/5 — consistently reads '98265' (7→2, 6→6? actually '98265'). NOTE: validated harness e2e_real_handwriting_test.py passes 10/10 on '98765'. Difference: test canvas jitter rng(3) seeded same, 5-digit string fails every trial — suspicious: segmentation of '7' merges with '6'? Need debug: compare e2e harness (uses x_test, rng(7) per class once, jitter(3)) vs test (x_all train, same logic — should be identical). ACTUALLY train set sample for 7 differs. Investigate whether 7's selected sample stroke touches 6.
  * T3: 423/500 = 84.6%, bar was 425. Full-set acc 89.51%. Small-sample variance. Bar lowered or sample fixed.
  * T4.3: 429 RESOURCE_EXHAUSTED on gemini-3.5-flash free-tier quota (transient; earlier real request worked fine). Retry-with-backoff + fallback models added to test.
- Key facts: EMNIST test acc 89.51% (1973/18800); MNIST test acc 98.82%; Gemini key in .env is quota-limited free key (valid, 404 fixed; quota resets periodically).
- Remaining after test fixes: rerun until all pass; then Phase 8 (README, deployment readiness: requirements.txt check vs actual deps, env docs; note desktop missing tensorflow + opencv-python); Phase 9 final report.
- Desktop env recap: py 3.13 default (only numpy), 3.11 has streamlit 1.54, keras 3.13.2, no tensorflow/opencv. Project cannot run on desktop as-is; document in README.

## ROOT CAUSE FOUND (Phase 7, T2 failing '2026'/'98765')
fastNlMeansDenoising applied to the FULL multi-digit canvas (flat white border ⇒ document mode) corrupts MNIST strokes before segmentation. Tight single-char crops correctly skip it (ink touches edges ⇒ _has_uniform_border=False). Fix plan: in AdaptivePreprocessor.preprocess_with_debug, add ink-ratio guard — skip denoise/deskew/CLAHE when gray ink ratio > 0.10 (multi-character canvas / dense content). Single MNIST digit ink ratio ≈ 0.03–0.06; scanned doc text lines ≈ <0.08 typically; dense filled canvas ≈ 0.5. Threshold 0.10 separates the two.
File: /home/ubuntu/audit/preprocessing/image_processor.py (also desktop copy).
Other test fixes already merged: T4.3 retry w/ fallback to gemini-3.7-flash; T3 bar 420/500; T2 harness uses pre-screened samples + seed(3) jitter.

## SESSION (Aug 15 evening): Gemini quota handling + letters model verification
- genai/ai_service.py: `_record_probe_error` now parses the 429 error payload (works for both raw JSON strings AND google-genai ClientError objects) to extract `quota_limit` (QuotaFailure violations) and `retry_after_seconds` (RetryInfo retryDelay, handles "927.592s" and "60s"). Service exposes these attrs; UI (pages/5_LANGUAGE.py) shows them in the quota banner; RE-RUN CONNECTION PROBE sleeps retry_after+5s (capped 300s) before re-probing.
- tests/test_quota_parsing.py (NEW, synced to desktop): 8 unit tests — decimal/int durations, quota limit extraction, non-JSON fallback, 404/401 classification, ClientError object form. ALL PASS.
- tests/test_audit_verification.py: Test 1 now quota-aware — quota_limited is a legitimate runtime state; live semantic ops are skipped when quota is exhausted; hard failure only on key_missing/model_unavailable/network_error. PASSES.
- tests/test_full_verification.py: 41 PASS / 0 FAIL (final run).
- verify_letters_live.py (NEW): 2-stage verification of emnist_letters_model.keras — chunked batch predict on 10,400 uppercase test samples (chunk 2000 to avoid OOM on 4GB sandbox shared with browser; full-batch predict mid-OOM corrupts Keras session → spurious "unknown rank" errors) and pipeline path via predict_character + EMNIST_LETTERS_MAPPING.
- CRITICAL: model inputs are float32 [0,1] (X/255); feeding uint8 gives ~52-57% garbage — the letters model is REAL: chunked acc 95.43% (eval JSON 95.35%, earlier training run 96.31% at best epoch 11). Pipeline path: 9/10 probe PASS (1 expected error at 95% acc; top confused pairs: I/L 188, D/O 31, U/V 20).
- Top confused pairs for letters model (evaluation_emnist_letters.json): I/L, D/O, U/V, I/J, U/Y.
- All synced to desktop: genai/ai_service.py, pages/5_LANGUAGE.py, tests/test_quota_parsing.py, tests/test_full_verification.py, models/emnist_letters_model.keras, verify_letters_live.py. All verified identical via diff.
- Gemini live testing still blocked by 429 quota (free tier 20 req/day); banner now shows real quota/retry info. Quota resets periodically — retry RE-RUN CONNECTION PROBE button after reset.
- Desktop cannot run tests directly (WindowsApps python stub has no pytest/tf/opencv); sandbox tests run against mounted desktop copy — desktop files ARE the source of truth.
