# BuildDelay AI — Agent Rules

Before editing:
1. Read PRD.md.
2. Check git status.
3. Inspect only files relevant to the task.

Source of truth:
1. Assignment PDF
2. PRD.md
3. Current implementation
4. Latest user decision
5. README.md

Core constraints:
- Keep RandomForestClassifier.
- Classes: Low / Moderate / High.
- Keep feature names/order consistent across generate_data.py, train_model.py, app.py.
- Confidence = predict_proba(), not real-world delay probability.
- Use synthetic data only and disclose that clearly.
- Do not invent model metrics.
- Use pathlib/relative paths.
- No paid services or generative-AI APIs.
- Keep code beginner-readable.
- Do not over-engineer.

Testing:
- Run only tests relevant to changed code first.
- Run full generate → train → app validation only when integration requires it.
- Never claim an unexecuted test passed.

Git:
- Never force push.
- Never reset --hard.
- Preserve unrelated changes.
- Commit only after meaningful validation.

UI work:
- Preserve ML behavior.
- Portfolio-quality construction/engineering dashboard.
- Functionality > explainability > assignment compliance > decoration.

Use Ponytail full principles:
- Reuse existing code first.
- Prefer stdlib/native/existing dependencies.
- Make the smallest safe diff.
- Keep explanations concise.