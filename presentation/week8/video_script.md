# Week 8 Individual Video: Narration Script

**Target length:** about 8 minutes (brief: 5–10). Read at a normal pace (~130 words a minute).
Rephrase in your own words wherever it feels stiff: the brief asks you to *personally explain*
your work, so it should sound like you, not like reading.

**Recording:** one audio file per slide, named `slide01.m4a` … `slide11.m4a` (mp3/wav also
fine), in `presentation/week8/recordings/`. Then run
`python presentation/week8/make_video.py` to build the MP4. A phone voice-memo app works.
Alternatively, present the PDF full-screen and record with Win + Alt + R (Xbox Game Bar).

| Slide | Brief section | Time |
|---|---|---|
| 1–2 | Introduction | 0:45 |
| 3–6 | Your track contribution | 2:45 |
| 7–8 | Cross-track collaboration | 1:45 |
| 5–6, 10 | Testing, refinement & final outcome | (covered in 5, 6, 10) |
| 9 | Overall HealthConnect solution | 1:00 |
| 10–11 | Limitations & closing | 1:30 |

---

### Slide 1: Title (≈15 s)
Hello, my name is Gyimah Ramsey Opoku, and I'm on the Machine Learning Engineering track of
the AnalystLab Africa Experience Lab. This is my Week 8 presentation for the HealthConnect
project: final ML pipeline integration and readiness.

### Slide 2: The problem and my role (≈35 s)
HealthConnect's question is how the clinic can use data and AI to reduce missed appointments
and improve patient support. My part of that is engineering. A model sitting in a notebook
can't be used by anyone else. Other tracks, and eventually a clinic system, need to score
patients the same way every time, and bad input has to be refused, not quietly scored. So my
responsibility was to turn models into a tested, reproducible scoring pipeline, integrate the
Data Science model, and prove it works, while being honest about what it can't do.

### Slide 3: Project journey (≈35 s)
In Week 6 I built the data pipeline over twenty-seven source tables, engineered seventy-five
features, and trained six models, each compared against a baseline. Only the High-Risk Patient
model showed real lift. In Week 7 I tested that model as a pipeline, and the Week 6 code passed
only two of fourteen reliability checks. After refining it, it passes all fourteen, with
sixty-one automated tests. This week, Week 8, was about final integration: reproducing it on
a second machine, proving the error handling, and integrating with Data Science.

### Slide 4: What I built (≈50 s)
Here's the pipeline. Input comes in as a CSV, a table, or a single patient. It's validated
first: every column is checked, categories are matched against a fixed list, numbers are
range-checked, and missing values are caught. Valid rows go into one saved scikit-learn
pipeline, which holds the encoding and the random forest together, so training and scoring
can't drift apart. That was the bug that broke Week 6. The output is a risk probability, a
flag, the threshold used and the model version, and every run is logged.
The tools were Python, pandas, scikit-learn, joblib and pytest, plus a command-line tool with
proper exit codes. The threshold lives in the model's metadata, so it can be changed without
retraining.

### Slide 5: Evidence (≈45 s)
These are the results. Fourteen of fourteen reliability checks, up from two. Sixty-one of
sixty-one automated tests, and when I deliberately injected five bugs, the tests caught all
five. Eleven of eleven error cases behave exactly as specified. And this week I re-cloned the
project on Windows with Python 3.12, a different setup from Week 7, retrained, and got an
identical model: same data fingerprint, same metrics, and zero patients whose flag changed.
On the held-out test set, ROC-AUC went from 0.727 to 0.721 after I removed leaked features, so
fixing the leakage cost almost nothing. At the 0.33 threshold, recall is 0.785, but 67.6
percent of patients get flagged, which I come back to under limitations.

### Slide 6: Bad input is refused (≈35 s)
This is what the error handling looks like. A valid patient is scored. An age of 250, a blank
age, the word "fifty", or an unknown visit type is rejected, with the CSV line and the
reason, and nothing is written. If you have a mixed batch, the skip-invalid option scores the
good rows and writes the bad ones to a rejects file. An age that's valid but outside the
training data is scored, with a warning in the log. And if the model file is missing, you get
a clear message telling you to run the training script.

### Slide 7: Data Science collaboration (≈55 s)
For cross-track collaboration I worked with Data Science, with Emmanuella Esi Arhin. She
handed me her final model package: an admission-proxy scorer, a requirements note, and fifteen
interface contract tests. She was clear that the model is a validated null result. It has no
predictive signal and must not be used for decisions, so what I was testing is the interface.
Before running anything, I reviewed her code and checked the model file. Then I ran her tests
in two environments, mine and one matching her training setup: fifteen of fifteen in both.
I then ran her same tests against my own pipeline: eleven of fifteen. And I fed all five
thousand rows from my data into her scorer. All were scored and none were rejected.

### Slide 8: What the collaboration revealed (≈50 s)
The four failures weren't bugs. They showed that our two scoring contracts disagree. Her
scorer imputes missing values and scores unknown categories with a warning, while mine rejects
them, and we handle empty input differently. We also have different limits, Age 110 versus
120, and decimals versus whole numbers. I asked her to confirm her prior-visits feature only
counts earlier visits, because that was exactly the leakage bug I fixed in Week 7. And since her
model needs the diagnosis, it can only run after consultation, so I won't connect it to
booking. I've sent her the results and a proposal for one shared contract, and the changes and
retest will follow her reply.

### Slide 9: How my work connects (≈60 s)
Here's how it fits together. Data Analytics provides the validated patterns and the valid
input ranges. Data Science provides the model, the target and the operating rule. My pipeline
turns that into a validated, reproducible scoring workflow, and Project Management gets the
issues, decisions and readiness status. The GenAI assistant handles patient admin questions
from the Knowledge Base and doesn't call the model.
The business value is a safe, repeatable way to score patients, so that once there's a model
with real signal, the clinic can test prioritising reminders and follow-up for patients likely
to miss appointments. My contribution means any model Data Science produces gets scored the
same way, with bad input refused and every run logged.

### Slide 10: Limitations (≈45 s)
I want to be clear about the limits. There's no proven predictor yet: the Data Science model
has no signal, with a hold-out AUC of 0.467, and my High-Risk label is a constructed rule, not
a real outcome. The data is synthetic, and only five thousand of sixty thousand appointments
have full context. The threshold isn't decided: 0.33 flags two-thirds of patients, which is
too many to follow up. It's a command-line tool and Python API, not a production web service.
And fairness by province or facility type can't be tested with this data. So it's ready to
demonstrate, not to make decisions about real patients.

### Slide 11: Closing (≈40 s)
My key takeaway: a model is only useful when other people can run it safely and get the same
answer. Testing showed the Week 6 code wasn't usable, and now it is, with evidence. What I
learned is to test the pipeline and not just the model, to report null results honestly, and
that collaboration needs agreed contracts, not just shared files. In a future version, I'd
retrain on real outcomes like attendance, agree one shared contract with Data Science, agree
the threshold with the clinic, and add a web API, monitoring and fairness checks.
Thank you for watching.
