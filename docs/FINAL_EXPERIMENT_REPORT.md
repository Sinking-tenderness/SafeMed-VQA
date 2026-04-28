# SafeMed-VQA++ Final Experiment Report

## 1. Project Goal

SafeMed-VQA++ is a medical multimodal safety system for visual question answering.  
The core objective is not simply to answer more questions, but to answer safely:

- answer when image evidence is sufficient;
- abstain when visual evidence is missing, degraded, occluded, or clinically unsafe;
- reduce high-confidence hallucination in medical VQA;
- make answer / abstain decisions explicit, structured, and evaluable.

The final system studies the full path:

```text
Base Model
  → SFT
  → SFT + Language/Visual Merger LoRA
  → Utility-DPO
  → Safety-DPO-v2
2. Data and Safety Setting

The training pipeline uses paired clean/degraded medical VQA samples.
Degraded samples simulate clinically relevant visual insufficiency, including:

center mask
local occlusion
border truncation
random crop
gaussian blur
speckle noise
resolution drop
contrast / brightness shift

Teacher labels provide structured supervision:

{
  "decision": "answer | abstain",
  "answer": "...",
  "abstain_type": "...",
  "risk_level": "...",
  "raw_confidence": 0.0,
  "explanation": "..."
}

This allows the student model to learn not only the answer content, but also the safety boundary between answer and abstain.

3. Method Overview
3.1 Base Model

The base model is Qwen3-VL-8B-Instruct.
It has general medical VQA ability, but does not reliably follow the project-specific safe JSON schema.

3.2 SFT

SFT teaches the model to produce structured JSON outputs with explicit decision, abstain type, risk level, answer, and explanation.

3.3 SFT + Visual Merger LoRA

The stronger SFT model additionally adapts language modules and visual merger modules:

language adapter tensors: 504
visual adapter tensors: 16
visual.merger adapter tensors: 4
visual.deepstack_merger_list adapter tensors: 12

This strengthens the connection between visual evidence and safe answer/abstain behavior.

3.4 Utility-DPO

The first DPO version used mixed preference pairs, including over-answer, wrong-answer, over-abstain, and invalid-schema cases.
It improved answer utility, but also introduced answer bias.

3.5 Safety-DPO-v2

Safety-DPO-v2 redesigned the preference data to focus on medical safety:

over_answer: 55
high_risk_wrong_answer: 35
invalid_schema: 5
over_abstain: 0

The goal is to reduce unsafe strong answering under insufficient evidence, rather than simply increasing answer rate.

4. Final Results
Model	JSON Success	Decision Match	Precise Answer	Precise Abstain	Over-Answer	Over-Abstain	Clean Match	Degraded Match	Paired Boundary	Answer Accuracy	Effective Answer Acc.
Base unstructured	0.0000	0.6707	0.6030	0.8554	0.1446	0.3970	null	null	0.3671	0.6206	0.3742
SFT LoRA vLLM	0.9911	0.8546	0.8848	0.7438	0.2479	0.1061	0.8973	0.8117	0.6962	0.6943	0.6144
SFT + Merger	0.9878	0.8620	0.8833	0.7645	0.2273	0.1030	0.9103	0.8135	0.7468	0.6895	0.6091
Utility-DPO	0.9823	0.8668	0.8924	0.7397	0.2438	0.0894	0.9116	0.8225	null	0.6910	0.6167
Safety-DPO-v2	0.9789	0.8686	0.8773	0.7769	0.2066	0.1000	0.9114	0.8262	0.7405	0.6978	0.6121
5. Main Findings
Finding 1: SFT gives the model structured safety behavior

Compared with the base model, SFT enables stable JSON output and explicit answer/abstain decisions.

Finding 2: Visual merger LoRA is a strong safety baseline

SFT + Merger improves clean/degraded decision stability and becomes the strongest non-DPO baseline.

Finding 3: Utility-DPO improves answer tendency but creates answer bias

Utility-DPO improves some answer-side metrics, but over-answer rate rises to 0.2438.
This indicates that DPO data design is critical: including too many over-abstain corrections can encourage the model to answer too aggressively.

Finding 4: Safety-DPO-v2 closes the safety loop

Safety-DPO-v2 reduces over-answer rate from 0.2273 to 0.2066 compared with SFT + Merger, while improving teacher decision match from 0.8620 to 0.8686 and degraded decision match from 0.8135 to 0.8262.

This shows that safety-oriented preference alignment can further improve medical VQA abstention boundaries.

6. Final Model Choice

The final recommended model is:

dpo_safety_v2_merger

The strongest stable baseline is:

sft_lora_merger_transformers

Utility-DPO is retained as an ablation showing that preference alignment can improve utility but may introduce answer bias if pair composition is not safety-oriented.

7. Final Story

The project demonstrates a complete safety alignment pipeline for medical multimodal VQA:

1. Build clean/degraded paired medical VQA samples.
2. Use a teacher model to label answer/abstain decisions.
3. Distill structured safety behavior into a Qwen3-VL-8B student.
4. Adapt language and visual merger modules with LoRA.
5. Evaluate answer correctness, abstention precision, over-answer risk, and paired boundary behavior.
6. Show that naive Utility-DPO can increase answer bias.
7. Redesign preference data as Safety-DPO-v2.
8. Reduce unsafe over-answering while preserving answer quality.

The final conclusion is:

Safety-DPO-v2 improves the medical safety boundary by reducing over-answer risk under insufficient visual evidence, while maintaining competitive answer correctness and overall decision alignment.
8. Limitations
JSON parse success slightly drops after Safety-DPO-v2.
Paired boundary success is slightly lower than SFT + Merger.
Teacher labels may contain noise.
The current DPO-v2 data is small, with 85 training pairs.
No clinician-in-the-loop human evaluation has been performed yet.
9. Future Work
Add more high-quality over-answer preference pairs.
Include schema-preserving DPO pairs to recover JSON stability.
Add confidence calibration for high-risk medical outputs.
Add clinician review for critical answer/abstain cases.
Expand evaluation to more medical modalities and datasets.
