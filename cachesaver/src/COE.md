# Common Error Types in LLMs

## Hallucination

An open challenge for LLMs is that they often hallucinate by making up facts or citing sources that do not exist (Li et al., 2023a; Zhang et al., 2023c). These hallucinated contents are often quite plausible-sounding, making it difficult even for humans to detect (Clark et al., 2021).

**Mitigation:**  
Several studies have proposed collecting automated feedback on potential factual inaccuracies by cross-referencing the model's output with credible knowledge sources. The gathered feedback can then be utilized by a subsequent refinement model to correct hallucinations.

**References:**
- Gao et al., 2023b
- Peng et al., 2023

---

## Unfaithful Reasoning

LLMs have exhibited a strong ability in solving complex reasoning tasks with improved strategies, such as Chain-of-Thought prompting (Wei et al., 2022b). However, recent studies (Golovneva et al., 2023; Ribeiro et al., 2023; Lyu et al., 2023b) found that LLMs occasionally make unfaithful reasoning, i.e., the derived conclusion does not follow the previously generated reasoning chain.

**Mitigation:**  
Existing works have proposed:
- Using automated feedback from external tools or models to guide the reasoning process (Xie et al., 2023; Yao et al., 2023a)
- Verifying the reasoning process and rectifying errors (He et al., 2023; Pan et al., 2023)
- Fine-tuning LLMs with process-based feedback (Huang et al., 2022; Lightman et al., 2023)

---

## Flawed Code

Besides generating natural language text, LLMs also show strong abilities to generate computer programs (i.e., code) (Chen et al., 2023b). However, the generated code can sometimes be flawed or incorrect.

**Mitigation:**  
Learning from automated feedback has been extensively applied in code generation (Chen et al., 2023d; Olausson et al., 2023), largely facilitated by the ease of obtaining such feedback through the execution of generated code with the corresponding compilers or interpreters.