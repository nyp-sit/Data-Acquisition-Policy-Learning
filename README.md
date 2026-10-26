# Data Acquisition & Policy Learning

**Specialist Diploma in Embodied AI · Competency Unit**
14-lesson delivery plan · Isaac Sim + SO-101

Seven lecture–lab pairs take beginners through the physical AI development cycle once for each method: reinforcement learning first, then imitation learning, then vision-language-action models. Every lab turns the previous lecture into something that runs, and every lesson opens by checking that learners have what the last one produced.

| | |
|---|---|
| **Lessons** | 14 × 3 h |
| **Pattern** | Lecture week → lab week |
| **Contact time** | 42 h (21 lecture + 21 lab) |
| **Assessment** | Test 1 · Test 2 · Assignment |
| **Simulation** | Isaac Sim 5.1 / Isaac Lab |
| **Hardware** | SO-101 leader + follower pods |

## Contents

1. [Planning assumptions](#1-planning-assumptions)
2. [Key focus areas and learning outcomes](#2-key-focus-areas-and-learning-outcomes)
   - [Learning outcomes](#learning-outcomes)
3. [Course structure](#3-course-structure)
   - [3.1 The learning spine](#31-the-learning-spine)
   - [3.2 Week-by-week timeline](#32-week-by-week-timeline)
   - [3.3 Lesson formats](#33-lesson-formats)
   - [3.4 How each lesson secures the last one](#34-how-each-lesson-secures-the-last-one)
   - [3.5 Coverage matrix](#35-coverage-matrix)

---

## 1 Planning assumptions

These assumptions come from the brief and the answers to the scheduling questions. If any of them changes, re-check the plan at the points noted.

- 14 lessons over 14 weeks, one 3-hour lesson per week, alternating: odd weeks are lectures, even weeks are labs. Each lecture is followed by the lab that puts it into practice.
- Test 1, Test 2 and the assignment presentation run in separate assessment slots outside the 14 lessons.
- Learners are adult beginners in robotics. They are assumed to have the SDAAI basics: Python, neural networks and supervised learning. No prior robotics, simulation or RL is assumed.
- One SO-101 leader + follower pair per pod of 3–4 learners, each pod with a front camera and a wrist camera. Every learner has a PC that can run Isaac Sim.
- Contact time is 42 h. If the unit must show 60 notional hours, the balance is guided self-study: assignment work, training runs between lessons, and pre-reading.

## 2 Key focus areas and learning outcomes

The brief's four key focus areas run through every lesson. Each one maps to the unit's learning outcomes.

| Code | Key focus | What it covers | Main LO |
|---|---|---|---|
| K1 | Concepts of physical AI development | The lifecycle (simulate → collect → train → evaluate → deploy → improve) and the three method families: RL, IL and VLA. | LO1–LO4 (context), LO3 |
| K2 | Data collection methods | Teleoperation, kinesthetic teaching (leader–follower, hand-guiding) and domain-randomised simulation. | LO1 |
| K3 | Control policy development | Reinforcement learning in Isaac Lab taken to the real arm, and imitation learning (ACT) from sim to arm. | LO2 |
| K4 | Evaluation and failure analysis | Task-specific benchmarks, success rates with confidence intervals, failure taxonomies and improvement loops. | LO4 |

### Learning outcomes

| LO | Statement |
|---|---|
| LO1 | Generate curated high-fidelity datasets using teleoperation, kinesthetic systems, and domain-randomised simulations for policy training. |
| LO2 | Develop control policies for embodied systems using imitation learning and reinforcement learning approaches. |
| LO3 | Integrate vision-language-action models and world action models for the orchestration of high-level planning and low-level execution of complex tasks. |
| LO4 | Evaluate policy performance and analyse failure modes against task-specific benchmarks across embodied systems. |

## 3 Course structure

### 3.1 The learning spine

The course has seven blocks, each a lecture week followed by a lab week. The order follows the brief: concepts, then data, then RL from simulation to the arm, then imitation learning from simulation to the arm with evaluation and deployment, then VLA. Each block depends on what the previous block produced.

| Block | Lessons | Weeks | Stage | Theme | Why it sits here |
|---|---|---|---|---|---|
| A | L1–L2 | 1–2 | Foundations | Physical AI and the toolchain | Learners need the vocabulary (joint, observation, action, policy) and hands-on time with both platforms before any method makes sense. |
| B | L3–L4 | 3–4 | Data | Collecting robot data three ways | Every method uses either a dataset or a randomised simulator. Both are built here and reused later. |
| C | L5–L6 | 5–6 | RL in sim | Reinforcement learning in Isaac Lab | RL comes first because it runs in simulation, uses the randomisation from Block B, and needs no dataset. |
| D | L7–L8 | 7–8 | RL sim → arm | Taking an RL policy to the real arm | Learners meet the reality gap for the first time. A reach policy transfers because everything it observes can be measured on the real arm. |
| E | L9–L10 | 9–10 | IL sim → arm | Imitation learning from demonstrations | Reuses the recording skills from Block B at scale. By now learners have seen RL, so the difference (a reward vs an answer key) is concrete. |
| F | L11–L12 | 11–12 | Evaluate & deploy | Deploying, evaluating and improving IL | The policy now works some of the time. Evaluation turns that into a number with a confidence interval and a fix to try. |
| G | L13–L14 | 13–14 | VLA & synthesis | Vision-language-action models | Uses the same demonstrations plus a sentence. The block ends by benchmarking every policy the pod has built. |

### 3.2 Week-by-week timeline

| Week | Lesson | Type | Title |
|---|---|---|---|
| 1 | L1 | Lecture | Physical AI: how a robot learns |
| 2 | L2 | Lab | Meet the tools: Isaac Sim and the SO-101 |
| 3 | L3 | Lecture | Robot data: teleoperation, kinesthetic and simulation |
| 4 | L4 | Lab | Collect data three ways |
| 5 | L5 | Lecture | Reinforcement learning I: trial and reward |
| 6 | L6 | Lab | Train a Reach policy in Isaac Lab |
| 7 | L7 | Lecture | Reinforcement learning II: sim to real |
| 8 | L8 | Lab | Run the Reach policy on the SO-101 |
| **—** | **T1** | **Assessment** | **Test 1 (L1–L8), separate slot** |
| 9 | L9 | Lecture | Imitation learning I: learning from demonstrations |
| 10 | L10 | Lab | Collect a quality dataset and train ACT |
| 11 | L11 | Lecture | Imitation learning II: deploy, evaluate, analyse failure |
| 12 | L12 | Lab | Deploy, evaluate and improve ACT |
| 13 | L13 | Lecture | Vision-language-action models and method choice |
| 14 | L14 | Lab | Run a VLA and benchmark the pod's policies |
| **—** | **T2** | **Assessment** | **Test 2 (L9–L14), separate slot** |
| **—** | **AS** | **Assessment** | **Assignment presentation, demo and Q&A, separate slot** |

### 3.3 Lesson formats

Both formats leave room for explanation and for learners to try things out. Beginners need the breaks and the recap time, so don't compress them.

#### Lecture (3 h)

| Time | Segment | Purpose |
|---|---|---|
| 0:00–0:15 | Entry check and recap | A short quiz on the previous lesson's gate items. Re-teach anything most of the class misses before moving on. |
| 0:15–1:00 | Part 1 | New concept, with a worked example. |
| 1:00–1:10 | Break | |
| 1:10–1:55 | Part 2 | New concept, with a live demo or video. |
| 1:55–2:05 | Break | |
| 2:05–2:40 | Part 3 | Concept that sets up the lab. |
| 2:40–3:00 | Activity and exit ticket | Pair activity, then a 3–5 question exit ticket. Preview the lab. |

#### Lab (3 h) with station rotation

| Time | Segment | Purpose |
|---|---|---|
| 0:00–0:15 | Briefing and safety | Recap the lecture idea the lab tests. Safety rules for the arm. Assign stations. |
| 0:15–1:25 | Rotation 1 | Group A at the Sim station (Isaac Sim, individual PCs). Group B at the Arm station (SO-101 pod). |
| 1:25–1:35 | Break | |
| 1:35–2:45 | Rotation 2 | Groups swap stations. |
| 2:45–3:00 | Debrief and gate sign-off | Pods show their gate evidence. The instructor signs it off or lists what must be finished before the next lecture. |

> **Note:** Some labs lean towards one station: L6 is mostly simulation, while L10 and L12 are mostly on the arm. Those lessons say so and give the other station analysis work.

### 3.4 How each lesson secures the last one

Three mechanisms make sure learners have what they need before new material starts:

- **Entry check** (first 15 minutes of every lesson): a short quiz or demonstration on the previous lesson's gate items. Anything most of the class misses is re-taught there and then.
- **Gate evidence** (end of every lesson): a named artefact or exit ticket. The instructor signs off lab gates, and learners must finish any open items before the next lecture.
- **Hand-forward artefacts:** most lab outputs feed a later lab, so a gap shows up straight away rather than at assessment time.

| Lesson | Needs from earlier | Produces | Used in |
|---|---|---|---|
| L1 | SDAAI basics | Vocabulary, lifecycle, SO-101 observation/action | L2 |
| L2 | L1 vocabulary | Calibration files, joint table, Isaac Sim fluency | L4, L6, L8 |
| L3 | L1–L2 | Dataset structure, quality rules, randomisation concepts | L4 |
| L4 | L3 + L2 calibration | Randomisation config, 10-episode dataset, quality log | L6, L10 |
| L5 | L4 randomisation | RL concepts, reward table, predictions | L6 |
| L6 | L5 + L2 joint table | Two Reach checkpoints, unit-mapping table | L8 |
| L7 | L6 results | Deployment contract, bridge loop, safety envelope, RL metrics | L8, L11 |
| L8 | L7 + L6 checkpoints | Sim vs real error table, RL failure log | Test 1, L11, L14 |
| L9 | L4 data + L8 RL contrast | BC/ACT concepts, pod dataset plan | L10 |
| L10 | L9 plan + L4 routine | 50-episode dataset, datasheet, ACT training | L12 |
| L11 | L10 ACT + L7 metrics | Evaluation protocol, CI method, failure taxonomy | L12 |
| L12 | L11 protocol + L10 ACT | Evaluation sheet, targeted data, two-task dataset | L14, Assignment |
| L13 | L9–L12 | VLA concepts, confusion-matrix test, method comparison | L14 |
| L14 | L13 + L12 + L8 | Confusion matrix, VLA vs ACT, pod benchmark | Test 2, Assignment |

### 3.5 Coverage matrix

● primary &nbsp; ◐ secondary

| Lesson | K1 Concepts | K2 Data | K3 Policies | K4 Evaluate | LO1 | LO2 | LO3 | LO4 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **L1** | ● | ◐ | ◐ | ◐ | ◐ | | ◐ | ◐ |
| **L2** | ● | ◐ | | | ● | | | |
| **L3** | ◐ | ● | | | ● | | | |
| **L4** | | ● | | ◐ | ● | | | ◐ |
| **L5** | ◐ | | ● | | | ● | | |
| **L6** | | ◐ | ● | | ◐ | ● | | |
| **L7** | | | ● | ◐ | | ● | | ◐ |
| **L8** | | | ● | ● | | ● | | ● |
| **L9** | ◐ | ◐ | ● | | ◐ | ● | | |
| **L10** | | ● | ● | | ● | ● | | |
| **L11** | | | ◐ | ● | | ◐ | | ● |
| **L12** | | ◐ | ◐ | ● | ◐ | ◐ | | ● |
| **L13** | ● | | | ◐ | | | ● | ◐ |
| **L14** | ◐ | | | ● | | | ● | ● |

---

**Not included in this README:** sections 4–8 of the full lesson plan document. They cover the 14 individual lesson plans, the assessment arrangement, the instructor preparation checklist, intentional omissions, changes from the 17-week draft, and the references checked.
