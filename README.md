# llm-alignment-survey

A curated reading list for large language model (LLM) alignment. Take a look at our new survey "[Large Language Model Alignment: A Survey](https://arxiv.org/abs/2309.15025)" on arXiv for more details!

Feel free to open an issue/PR or e-mail <thshen@tju.edu.cn> and <dyxiong@tju.edu.cn> if you find any missing areas, papers, or datasets. We will keep updating this list and survey.

If you find our survey useful, please kindly cite our paper:

```bibtex
@article{shen2023alignment,
      title={Large Language Model Alignment: A Survey}, 
      author={Shen, Tianhao and Jin, Renren and Huang, Yufei and Liu, Chuang and Dong, Weilong and Guo, Zishan and Wu, Xinwei and Liu, Yan and Xiong, Deyi},
      journal={arXiv preprint arXiv:2309.15025},
      year={2023}
}
```

## Table of Contents
- [llm-alignment-survey](#llm-alignment-survey)
  - [Related Surveys](#related-surveys)
  - [Why LLM Alignment?](#why-llm-alignment)
    - [LLM-Generated Content](#llm-generated-content)
      - [Undesirable Content](#undesirable-content)
      - [Unfaithful Content](#unfaithful-content)
      - [Malicious Uses](#malicious-uses)
      - [Negative Impacts on Society](#negative-impacts-on-society)
    - [Potential Risks Associated with Advanced LLMs](#potential-risks-associated-with-advanced-llms)
  - [What is LLM Alignment?](#what-is-llm-alignment)
  - [Outer Alignment](#outer-alignment)
    - [Non-recursive Oversight](#non-recursive-oversight)
      - [RL-based Methods](#rl-based-methods)
      - [SL-based Methods](#sl-based-methods)
    - [Scalable Oversight](#scalable-oversight)
  - [Inner Alignment](#inner-alignment)
  - [Mechanistic Interpretability](#mechanistic-interpretability)
  - [Attacks on Aligned Language Models](#attacks-on-aligned-language-models)
    - [Privacy Attacks](#privacy-attacks)
    - [Backdoor Attacks](#backdoor-attacks)
    - [Adversarial Attacks](#adversarial-attacks)
  - [Alignment Evaluation](#alignment-evaluation)
    - [Factuality Evaluation](#factuality-evaluation)
    - [Ethics Evaluation](#ethics-evaluation)
    - [Toxicity Evaluation](#toxicity-evaluation)
      - [Task-specific Evaluation](#task-specific-evaluation)
      - [LLM-centered Evaluation](#llm-centered-evaluation)
    - [Stereotype and Bias Evaluation](#stereotype-and-bias-evaluation)
      - [Task-specific Evaluation](#task-specific-evaluation-1)
      - [LLM-centered Evaluation](#llm-centered-evaluation-1)
      - [Hate Speech Detection](#hate-speech-detection)
    - [General Evaluation](#general-evaluation)


## Related Surveys
1. **Aligning Large Language Models with Human: A Survey.** Yufei Wang et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2307.12966)]
2. **Trustworthy LLMs: a Survey and Guideline for Evaluating Large Language Models' Alignment.** Yang Liu et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2308.05374)]
3. **Bridging the Gap: A Survey on Integrating (Human) Feedback for Natural Language Generation.** Patrick Fernandes et al. arXiv 2023. [[paper](https://arxiv.org/abs/2305.00955)]
4. **Augmented Language Models: a Survey.** Grégoire Mialon et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2302.07842)]
5. **An Overview of Catastrophic AI Risks.** Dan Hendrycks et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2306.12001)]
6. **A Survey of Large Language Models.** Wayne Xin Zhao et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2303.18223)]
7. **A Survey on Universal Adversarial Attack.** Chaoning Zhang et al. IJCAI 2021. [[Paper](https://arxiv.org/abs/2103.01498)]
8. **Survey of Hallucination in Natural Language Generation.** Ziwei Ji et al. ACM Computing Surveys 2022. [[Paper](https://arxiv.org/abs/2202.03629)]
9. **Automatically Correcting Large Language Models: Surveying the landscape of diverse self-correction strategies.** Liangming Pan et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2308.03188)]
10. **Automatic Detection of Machine Generated Text: A Critical Survey.** Ganesh Jawahar et al. COLING 2020. [[Paper](https://arxiv.org/abs/2011.01314)]

## Why LLM Alignment?

1. **Synchromesh: Reliable Code Generation from Pre-trained Language Models.** Gabriel Poesia et al. ICLR 2022. [[Paper](https://openreview.net/forum?id=KmtVD97J43e)]
2. **LLM-Planner: Few-Shot Grounded Planning for Embodied Agents with Large Language Models.** Chan Hee Song et al. ICCV 2023. [[Paper](https://arxiv.org/abs/2212.04088)]
3. **Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents.** Wenlong Huang et al. PMLR 2022. [[Paper](https://proceedings.mlr.press/v162/huang22a.html)]
4. **Tool Learning with Foundation Models.** Yujia Qin et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2304.08354)]
5. **Ethical and social risks of harm from Language Models.** Laura Weidinger et al. arXiv 2021. [[Paper](https://arxiv.org/abs/2112.04359)]

### LLM-Generated Content

#### Undesirable Content

1. **Predictive Biases in Natural Language Processing Models: A Conceptual Framework and Overview.** Deven Shah et al. arXiv 2019. [[Paper](https://arxiv.org/abs/1912.11078)]
2. **RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models.** Samuel Gehman et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2009.11462)]
3. **Extracting Training Data from Large Language Models.** Nicholas Carlini et al. arXiv 2012. [[Paper](https://arxiv.org/abs/2012.07805)]
4. **StereoSet: Measuring stereotypical bias in pretrained language models.** Moin Nadeem et al. arXiv 2020. [[Paper](https://arxiv.org/abs/2004.09456)]
5. **CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models.** Nikita Nangia et al. EMNLP 2020. [[Paper](https://arxiv.org/abs/2010.00133)]
6. **HONEST: Measuring Hurtful Sentence Completion in Language Models.** Debora Nozza et al. NAACL 2021. [[Paper](https://aclanthology.org/2021.naacl-main.191)]
7. **Language Models are Few-Shot Learners.** Tom Brown et al. NeurIPS 2020. [[Paper](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html)]
8. **Persistent Anti-Muslim Bias in Large Language Models.** Abubakar Abid et al. AIES 2021. [[Paper](https://dl.acm.org/doi/10.1145/3461702.3462624)]
9. **Gender and Representation Bias in GPT-3 Generated Stories.** Li Lucy et al. WNU 2021. [[Paper](https://aclanthology.org/2021.nuse-1.5/)]

#### Unfaithful Content

1. **Measuring and Improving Consistency in Pretrained Language Models.** Yanai Elazar et al. TACL 2021. [[Paper](https://aclanthology.org/2021.tacl-1.60.pdf)]
2. **GPT-3 Creative Fiction.** Gwern. 2023. [[Blog](https://gwern.net/gpt-3)]
3. **GPT-3: What’s It Good for?** Robert Dale. Natural Language Engineering 2020. [[Paper](https://www.cambridge.org/core/journals/natural-language-engineering/article/gpt3-whats-it-good-for/0E05CFE68A7AC8BF794C8ECBE28AA990)]
4. **Scaling Language Models: Methods, Analysis & Insights from Training Gopher.** Jack W. Rae et al. arXiv 2021. [[Paper](https://arxiv.org/abs/2112.11446)]
5. **TruthfulQA: Measuring How Models Mimic Human Falsehoods.** Stephanie Lin et al. ACL 2022. [[Paper](https://arxiv.org/abs/2109.07958)]
6. **Towards Tracing Knowledge in Language Models Back to the Training Data.** Ekin Akyurek et al. EMNLP 2020. [[Paper](https://aclanthology.org/2022.findings-emnlp.180/)]
7. **Sparks of Artificial General Intelligence: Early experiments with GPT-4.** Sébastien Bubeck et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2303.12712)]
8. **Navigating the Grey Area: Expressions of Overconfidence and Uncertainty in Language Models.** Kaitlyn Zhou et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2302.13439)]
9. **Patient and Consumer Safety Risks When Using Conversational Assistants for Medical Information: An Observational Study of Siri, Alexa, and Google Assistant.** Reza Asadi et al. 2018. [[Paper](https://www.academia.edu/87073068/Patient_and_Consumer_Safety_Risks_When_Using_Conversational_Assistants_for_Medical_Information_An_Observational_Study_of_Siri_Alexa_and_Google_Assistant)]
10. **Will ChatGPT Replace Lawyers?** Kate Rattray. 2023. [[Blog](https://www.clio.com/blog/chat-gpt-lawyers)]
11. **Constitutional AI: Harmlessness from AI Feedback.** Yuntao Bai et al. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.08073)]

#### Malicious Uses

1. **Truth, Lies, and Automation How Language Models Could Change Disinformation.** Ben Buchanan et al. Center for Security and Emerging Technology, 2021. [[Paper](https://cset.georgetown.edu/publication/truth-lies-and-automation/)]
2. **Understanding the Capabilities, Limitations, and Societal Impact of Large Language Models.** Alex Tamkin et al. arXiv 2021. [[Paper](https://arxiv.org/abs/2102.02503)]
3. **Deal or No Deal? End-to-End Learning for Negotiation Dialogues.** Mike Lewis et al. arXiv 2017. [[Paper](https://arxiv.org/abs/1706.05125)]
4. **Evaluating Large Language Models Trained on Code.** Anne-Laure Ligozat et al. arXiv 2021. [[Paper](https://arxiv.org/abs/2107.03374)]
5. **Artificial intelligence and biological misuse: Differentiating risks of language models and biological design tools.** Jonas B. Sandbrink. arXiv 2023. [[Paper](https://arxiv.org/abs/2306.13952)]

#### Negative Impacts on Society

1. **Sustainable AI: AI for sustainability and the sustainability of AI. Aimee van Wynsberghe.** AI and Ethics 2021. [[Paper](https://link.springer.com/article/10.1007/s43681-021-00043-6)]
2. **Unraveling the Hidden Environmental Impacts of AI Solutions for Environment.** Anne-Laure Ligozat et al. arXiv 2021. [[Paper](https://arxiv.org/abs/2110.11822)]
3. **GPTs are GPTs: An Early Look at the Labor Market Impact Potential of Large Language Models.** Tyna Eloundou et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2303.10130)]

### Potential Risks Associated with Advanced LLMs

1. **Formalizing Convergent Instrumental Goals.** Tsvi Benson-Tilsen et al. AAAI AIES Workshop 2016. [[Paper](https://www.semanticscholar.org/paper/Formalizing-Convergent-Instrumental-Goals-Benson-Tilsen-Soares/d7b321d8d88381a2a84e9d6e8f8f34ee2ed65df2)]
2. **Model evaluation for extreme risks.** Toby Shevlane et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15324)]
3. **Aligning AI Optimization to Community Well-Being.** Stray J. International Journal of Community Well-being 2020. [[Paper](https://europepmc.org/article/MED/34723107)]
4. **What are you optimizing for? Aligning Recommender Systems with Human Values.** Jonathan Stray et al. ICML 2020. [[Paper](https://arxiv.org/abs/2107.10939)]
5. **Model evaluation for extreme risks.** Toby Shevlane et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.15324)]
6. **Human-level play in the game of Diplomacy by combining language models with strategic reasoning.** Meta Fundamental AI Research Diplomacy Team (FAIR) et al. Science 2022. [[Paper](https://vlgiitr.github.io/papers_we_read/summaries/CICERO.html)]
7. **Characterizing Manipulation from AI Systems.** Micah Carroll et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2303.09387)]
8. **Deceptive Alignment Monitoring.** Andres Carranza et al. ICML AdvML Workshop 2023. [[Paper](https://arxiv.org/abs/2307.10569)]
9. **The Superintelligent Will: Motivation and Instrumental Rationality in Advanced Artificial Agents.** Nick Bostrom. Minds and Machines 2012. [[Paper](https://link.springer.com/article/10.1007/s11023-012-9281-3)]
10. **Is Power-Seeking AI an Existential Risk?** Joseph Carlsmith. arXiv 2023. [[Paper](https://arxiv.org/abs/2206.13353)]
11. **Optimal Policies Tend To Seek Power.** Alexander Matt Turner et al. NeurIPS 2021. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/c26820b8a4c1b3c2aa868d6d57e14a79-Paper.pdf)]
12. **Parametrically Retargetable Decision-Makers Tend To Seek Power.** Alexander Matt Turner et al. NeurIPS 2022. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/cb3658b9983f677670a246c46ece553d-Paper-Conference.pdf)]
13. **Power-seeking can be probable and predictive for trained agents.** Victoria Krakovna et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2304.06528)]
14. **Discovering Language Model Behaviors with Model-Written Evaluations.** Ethan Perez et al. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.09251)]

## What is LLM Alignment?

1. **Some Moral and Technical Consequences of Automation: As Machines Learn They May Develop Unforeseen Strategies at Rates That Baffle Their Programmers.** Norbert Wiener. Science 1960. [[Paper](https://doi.org/10.1126/science.131.3410.1355)]
2. **Coherent Extrapolated Volition.** Eliezer Yudkowsky. Singularity Institute for Artificial Intelligence 2004. [[Paper](https://intelligence.org/files/CEV.pdf)]
3. **The Basic AI Drives.** Stephen M. Omohundro. AGI 2008. [[Paper](https://dl.acm.org/doi/10.5555/1566174.1566226)]
4. **The Superintelligent Will: Motivation and Instrumental Rationality in Advanced Artificial Agents.** Nick Bostrom. Minds and Machines 2012. [[Paper](https://doi.org/10.1007/s11023-012-9281-3)]
5. **General Purpose Intelligence: Arguing the Orthogonality Thesis.** Stuart Armstrong. Analysis and Metaphysics 2013. [[Paper](https://www.fhi.ox.ac.uk/wp-content/uploads/Orthogonality_Analysis_and_Metaethics-1.pdf)]
6. **Aligning Superintelligence with Human Interests: An Annotated Bibliography.** Nate Soares. Intelligence 2015. [[Paper](http://intelligence.org/files/AnnotatedBibliography.pdf)]
7. **Concrete Problems in AI Safety.** Dario Amodei et al. arXiv 2016. [[Paper](https://arxiv.org/abs/1606.06565)]
8.  **The Mythos of Model Interpretability.** Zachary C. Lipton. arXiv 2017. [[Paper](https://arxiv.org/abs/1606.03490)]
9.  **AI Safety Gridworlds.** Jan Leike et al. arXiv 2017. [[Paper](https://arxiv.org/abs/1711.09883)]
10. **Overview of Current AI Alignment Approaches.** Micah Carroll. 2018. [[Paper](https://micahcarroll.github.io/assets/ValueAlignment.pdf)]
11. **Risks from Learned Optimization in Advanced Machine Learning Systems.** Evan Hubinger et al. arXiv 2019. [[Paper](https://arxiv.org/abs/1906.01820)]
12. **An Overview of 11 Proposals for Building Safe Advanced AI.** Evan Hubinger. arXiv 2020. [[Paper](https://arxiv.org/abs/2012.07532)]
13. **Unsolved Problems in ML Safety.** Dan Hendrycks et al. arXiv 2021. [[Paper](https://arxiv.org/abs/2109.13916)]
14. **A Mathematical Framework for Transformer Circuits.** Nelson Elhage et al. Transformer Circuits Thread 2021. [[Paper](https://transformer-circuits.pub/2021/framework/index.html)]
15. **Alignment of Language Agents.** Zachary Kenton et al. arXiv 2021. [[Paper](https://arxiv.org/abs/2103.14659)]
16. **A General Language Assistant as a Laboratory for Alignment.** Amanda Askell et al. arXiv 2021. [[Paper](https://arxiv.org/abs/2112.00861)]
17. **A Transparency and Interpretability Tech Tree.** Evan Hubinger. 2022. [[Blog](https://www.lesswrong.com/posts/nbq2bWLcYmSGup9aF/a-transparency-and-interpretability-tech-tree)]
18. **Understanding AI Alignment Research: A Systematic Analysis.** J. Kirchner et al. arXiv 2022. [[Paper](https://arxiv.org/abs/2206.02841)]
19. **Softmax Linear Units.** Nelson Elhage et al. Transformer Circuits Thread 2022. [[Paper](https://transformer-circuits.pub/2022/solu/index.html)]
20. **The Alignment Problem from a Deep Learning Perspective.** Richard Ngo. arXiv 2022. [[Paper](https://arxiv.org/abs/2209.00626)]
21. **Paradigms of AI Alignment: Components and Enablers.** Victoria Krakovna. 2022. [[Blog](https://www.lesswrong.com/posts/JC7aJZjt2WvxxffGz/paradigms-of-ai-alignment-components-and-enablers)]
22. **Progress Measures for Grokking via Mechanistic Interpretability.** Neel Nanda et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2301.05217)]
23. **Agentized LLMs Will Change the Alignment Landscape.** Seth Herd. 2023. [[Blog](https://www.lesswrong.com/posts/dcoxvEhAfYcov2LA6/agentized-llms-will-change-the-alignment-landscape)]
24. **Language Models Can Explain Neurons in Language Models.** Steven Bills et al. 2023. [[Paper](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)]
25. **Core Views on AI Safety: When, Why, What, and How.** Anthropic. 2023. [[Blog](https://www.anthropic.com/index/core-views-on-ai-safety)]

## Outer Alignment
### Non-recursive Oversight
#### RL-based Methods

1.  **Proximal Policy Optimization Algorithms.** John Schulman et al. arXiv 2017. [[Paper](https://arxiv.org/abs/1707.06347)]
2.  **Fine-Tuning Language Models from Human Preferences.** Daniel M Ziegler et al. arXiv 2019. [[Paper](https://arxiv.org/abs/1909.08593)]
3.  **Learning to Summarize with Human Feedback.** Nisan Stiennon et al. NeurIPS 2020. [[Paper](https://proceedings.neurips.cc/paper/2020/file/1f89885d556929e98d3ef9b86448f951-Paper.pdf)]
4.  **Training Language Models to Follow Instructions with Human Feedback.** Long Ouyang et al. NeurIPS 2022. [[Paper](https://arxiv.org/abs/2203.02155)]
5.  **Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback.** Yuntao Bai et al. arXiv 2022. [[Paper](https://arxiv.org/abs/2204.05862)]
6.  **RL4F: Generating Natural Language Feedback with Reinforcement Learning for Repairing Model Outputs.** Afra Feyza Akyürek et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.08844)]
7.  **Improving Language Models with Advantage-Based Offline Policy Gradients.** Ashutosh Baheti et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.14718)]
8.  **Scaling Laws for Reward Model Overoptimization.** Leo Gao et al. ICML 2023. [[Paper](https://proceedings.mlr.press/v202/gao23h/gao23h.pdf)]
9.  **Improving Alignment of Dialogue Agents via Targeted Human Judgements.** Amelia Glaese et al. arXiv 2022. [[Paper](https://arxiv.org/abs/2209.14375)]
10. **Aligning Language Models with Preferences through F-Divergence Minimization.** Dongyoung Go et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2302.08215)]
11. **Aligning Large Language Models through Synthetic Feedback.** Sungdong Kim et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.13735)]
12. **RLHF.** Ansh Radhakrishnan. Lesswrong 2022. [[Paper](https://www.lesswrong.com/posts/rQH4gRmPMJyjtMpTn/rlhf)]
13. **Guiding Large Language Models via Directional Stimulus Prompting.** Zekun Li et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2302.11520)]
14. **Aligning Generative Language Models with Human Values.** Ruibo Liu et al. NAACL 2022 Findings. [[Paper](https://aclanthology.org/2022.findings-naacl.18)]
15. **Second Thoughts Are Best: Learning to Re-Align with Human Values from Text Edits.** Ruibo Liu et al. NeurIPS 2022. [[Paper](https://openreview.net/pdf?id=u6OfmaGIya1)]
16. **Secrets of RLHF in Large Language Models Part I: PPO.** Rui Zheng et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2307.04964)]
17. **Principled Reinforcement Learning with Human Feedback from Pairwise or K-Wise Comparisons.** Banghua Zhu et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2301.11270)]
18. **Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback.** Stephen Casper et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2307.15217)]

#### SL-based Methods

1.  **Self-Diagnosis and Self-Debiasing: A Proposal for Reducing Corpus-Based Bias in NLP.** Timo Schick et al. TACL, 2021. [[Paper](https://aclanthology.org/2021.tacl-1.84)]
2.  **The Cringe Loss: Learning What Language Not to Model.** Leonard Adolphs et al. arXiv 2022. [[Paper](https://arxiv.org/abs/2211.05826)]
3.  **Leashing the Inner Demons: Self-detoxification for Language Models.** Canwen Xu et al. AAAI 2022. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/21406/21155)]
4.  **Calibrating Sequence Likelihood Improves Conditional Language Generation.** Yao Zhao et al. arXiv 2022. [[Paper](https://arxiv.org/abs/2210.00045)]
5.  **RAFT: Reward Ranked Finetuning for Generative Foundation Model Alignment.** Hanze Dong et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2304.06767)]
6.  **Chain of Hindsight Aligns Language Models with Feedback.** Hao Liu et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2302.02676)]
7.  **Training Socially Aligned Language Models in Simulated Human Society.** Ruibo Liu et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16960)]
8.  **Direct Preference Optimization: Your Language Model Is Secretly a Reward Model.** Rafael Rafailov et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18290)]
9.  **Training Language Models with Language Feedback at Scale.** Jérémy Scheurer et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2303.16755)]
10. **Preference Ranking Optimization for Human Alignment.** Feifan Song et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2306.17492)]
11. **RRHF: Rank Responses to Align Language Models with Human Feedback without Tears.** Zheng Yuan et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2304.05302)]
12. **SLiC-HF: Sequence Likelihood Calibration with Human Feedback.** Yao Zhao et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.10425)]
13. **LIMA: Less Is More for Alignment.** Chunting Zhou et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.11206)]


### Scalable Oversight

1.  **Supervising Strong Learners by Amplifying Weak Experts.** Paul Christiano et al. arXiv 2018. [[Paper](https://arxiv.org/abs/1810.08575)]
2.  **Scalable Agent Alignment via Reward Modeling: A Research Direction.** Jan Leike et al. arXiv 2018. [[Paper](https://arxiv.org/abs/1811.07871)]
3.  **AI Safety Needs Social Scientists.** Geoffrey Irving, and Amanda Askell. Distill 2019. [[Paper](https://distill.pub/2019/ai-safety-needs-social-scientists/)]
4.  **Learning to Summarize with Human Feedback.** Nisan Stiennon et al. NeurIPS 2020. [[Paper](https://proceedings.neurips.cc/paper/2020/hash/52b6c8e5a34e5e7e11e466a3d508d6a5-Abstract.html)]
5.  **Task Decomposition for Scalable Oversight (AGISF Distillation).** Charbel-Raphaël Segerie. 2023. [[Blog](https://www.lesswrong.com/posts/FFz6H35Gy6BArHxkc/task-decomposition-for-scalable-oversight-agisf-distillation)]
6.  **Measuring Progress on Scalable Oversight for Large Language Models.** Samuel R Bowman et al. arXiv 2022. [[Paper](https://arxiv.org/abs/2211.03540)]
7.  **Constitutional AI: Harmlessness from AI Feedback.** Yuntao Bai et al. CoRR 2022. [[Paper](https://arxiv.org/abs/2212.08073)]
8.  **Improving Factuality and Reasoning in Language Models through Multiagent Debate.** Yilun Du et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.14325)]
9.  **Evaluating Superhuman Models with Consistency Checks.** Lukas Fluri et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2306.09983)]
10. **AI Safety via Debate.** Geoffrey Irving et al. arXiv 2018. [[Paper](https://arxiv.org/abs/1805.00899)]
11. **AI Safety via Market Making.** Evan Hubinger. 2020. [[Blog](https://www.lesswrong.com/posts/YWwzccGbcHMJMpT45/ai-safety-via-market-making)]
12. **Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate.** Tian Liang et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.19118)]
13. **Let's Verify Step by Step.** Hunter Lightman et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.20050)]
14. **Introducing Superalignment.** OpenAI. 2023. [[Blog](https://openai.com/blog/introducing-superalignment)]
15. **Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision.** Zhiqing Sun et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.03047)]

## Inner Alignment

1. **Risks from Learned Optimization in Advanced Machine Learning Systems.** Evan Hubinger et al. arXiv 2021. [[Paper](https://arxiv.org/abs/1906.01820)]
2. **Goal Misgeneralization in Deep Reinforcement Learning.** Lauro Langosco et al. ICML 2022. [[Paper](https://arxiv.org/abs/2105.14111)]
3. **Goal Misgeneralization: Why Correct Specifications Aren't Enough For Correct Goals.** Rohin Shah et al. arXiv 2022. [[Paper](https://arxiv.org/abs/2210.01790)]
4. **Defining capability and alignment in gradient descent.** Edouard Harris. Lesswrong 2020. [[Blog](https://www.lesswrong.com/posts/Xg2YycEfCnLYrCcjy/defining-capability-and-alignment-in-gradient-descent)]
5. **Categorizing failures as “outer” or “inner” misalignment is often confused.** Rohin Shah. Lesswrong 2023. [[Blog](https://www.lesswrong.com/posts/JKwrDwsaRiSxTv9ur/categorizing-failures-as-outer-or-inner-misalignment-is)]
6. **Inner Alignment Failures" Which Are Actually Outer Alignment Failures.** John Wentworth. Lesswrong 2020. [[Blog](https://www.lesswrong.com/posts/HYERofGZE6j9Tuigi/inner-alignment-failures-which-are-actually-outer-alignment)]
7. **Relaxed adversarial training for inner alignment.** Evan Hubinger. Lesswrong 2019. [[Blog](https://www.lesswrong.com/posts/9Dy5YRaoCxH9zuJqa/relaxed-adversarial-training-for-inner-alignment)]
8. **The Inner Alignment Problem.** Evan Hubinger et al. Lesswrong 2019. [[Blog](https://www.lesswrong.com/posts/pL56xPoniLvtMDQ4J/the-inner-alignment-problem)]
9. **Three scenarios of pseudo-alignment.** Eleni Angelou. Lesswrong 2022. [[Blog](https://www.lesswrong.com/posts/W5nnfgWkCPxDvJMpe/three-scenarios-of-pseudo-alignment)]
10. **Deceptive Alignment.** Evan Hubinger et al. Lesswrong 2019. [[Blog](https://www.lesswrong.com/s/r9tYkB2a8Fp4DN8yB/p/zthDPAjh9w6Ytbeks)]
11. **What failure looks like.** Paul Christiano. AI Alignment Forum 2019. [[Blog](https://www.alignmentforum.org/posts/HBxe6wdjxK239zajf/more-realistic-tales-of-doom)]
12. **Concrete experiments in inner alignment.** Evan Hubinger. Lesswrong 2019. [[Blog](https://www.lesswrong.com/posts/uSdPa9nrSgmXCtdKN/concrete-experiments-in-inner-alignment)]
13. **A central AI alignment problem: capabilities generalization, and the sharp left turn.** Nate Soares. Lesswrong 2022. [[Blog](https://www.lesswrong.com/posts/GNhMPAWcfBCASy8e6/a-central-ai-alignment-problem-capabilities-generalization)]
14. **Clarifying the confusion around inner alignment.** Rauno Arike. AI Alignment Forum 2022. [[Blog](https://www.alignmentforum.org/posts/xdtNd8xCdzpgfnGme/clarifying-the-confusion-around-inner-alignment)]
15. **2-D Robustness.** Vladimir Mikulik. AI Alignment Forum 2019. [[Blog](https://www.alignmentforum.org/posts/2mhFMgtAjFJesaSYR/2-d-robustness)]
16. **Monitoring for deceptive alignment.** Evan Hubinger. Lesswrong 2022. [[Blog](https://www.lesswrong.com/posts/Km9sHjHTsBdbgwKyi/monitoring-for-deceptive-alignment)]

## Mechanistic Interpretability

1. **Notions of explainability and evaluation approaches for explainable artificial intelligence.** Giulia Vilone et al. arXiv 2020. [[Paper](https://www.sciencedirect.com/science/article/pii/S1566253521001093)]
2. **A Comprehensive Mechanistic Interpretability Explainer Glossary.** Neel Nanda. 2022. [[Paper](https://www.lesswrong.com/posts/vnocLyeWXcAxtdDnP/a-comprehensive-mechanistic-interpretability-explainer-and)]
3. **The Mythos of Model Interpretability.** Zachary C. Lipton. arXiv 2017. [[Paper](https://arxiv.org/abs/1606.03490)]
4. **AI research considerations for human existential safety (ARCHES).** Andrew Critch et al. arXiv 2020. [[Paper](https://arxiv.org/abs/2006.04948)]
5. **Concrete problems for autonomous vehicle safety: Advantages of Bayesian deep learning.** RT McAllister et al. IJCAI 2017. [[Paper](https://dl.acm.org/doi/10.5555/3171837.3171951)]
6. **In-context Learning and Induction Heads.** Catherine Olsson et al. Transformer Circuits Thread, 2022. [[Paper](https://arxiv.org/abs/2209.11895)]
7. **Transformer Feed-Forward Layers Are Key-Value Memories.** Mor Geva et al. EMNLP 2021. [[Paper](https://arxiv.org/abs/2012.14913)]
8. **Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space.** Mor Geva et al. EMNLP 2022. [[Paper](https://arxiv.org/abs/2203.14680)]
9. **Softmax Linear Units.** Nelson Elhage et al. Transformer Circuits Thread 2022. [[Paper](https://transformer-circuits.pub/2022/solu/index.html)]
10. **Toy Models of Superposition.** Nelson Elhage et al. Transformer Circuits Thread 2022. [[Paper](https://arxiv.org/abs/2209.10652)]
11. **Mechanistic Interpretability, Variables, and the Importance of Interpretable Bases.** Chris Olah. 2022. [[Paper](https://transformer-circuits.pub/2022/mech-interp-essay/index.html)]
12. **Knowledge Neurons in Pretrained Transformers.** Dai Damai et al. ACL 2021. [[Paper](https://aclanthology.org/2022.acl-long.581/)]
13. **Locating and editing factual associations in GPT.** Kevin Meng et al. NeurIPS 2022. [[Paper](https://arxiv.org/abs/2202.05262)]
14. **Eliciting Truthful Answers from a Language Model.** Kenneth Li et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2306.03341)]
15. **LEACE: Perfect linear concept erasure in closed form.** Nora Belrose et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2306.03819)]

## Attacks on Aligned Language Models

### Privacy Attacks

1. **Jailbreaker: Automated Jailbreak Across Multiple Large Language Model Chatbots.** Gelei Deng et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2307.08715)]
2. **Multi-step Jailbreaking Privacy Attacks on ChatGPT.** Haoran Li et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2304.05197)]

### Backdoor Attacks

1. **Prompt Injection Attack Against LLM-integrated Applications.** Yi Liu et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2306.05499)]
2. **Prompt as Triggers for Backdoor Attack: Examining the Vulnerability in Language Models.** Shuai Zhao et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.01219)]
3. **More Than You've Asked for: A Comprehensive Analysis of Novel Prompt Injection Threats to Application-Integrated Large Language Models.** Kai Greshake et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2302.12173)]
4. **Backdoor Attacks for In-Context Learning with Language Models.** Nikhil Kandpal et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2307.14692)]
5. **BadGPT: Exploring Security Vulnerabilities of ChatGPT via Backdoor Attacks to InstructGPT.** Jiawen Shi et al. arXiv 2023. [[Paper](https://arxiv.org/pdf/2304.12298.pdf)]

### Adversarial Attacks

1. **Universal and Transferable Adversarial Attacks on Aligned Language Models.** Andy Zou et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2307.15043)]
2. **Are Aligned Neural Networks Adversarially Aligned?.** Nicholas Carlini et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2306.15447)]
3. **Visual Adversarial Examples Jailbreak Large Language Models.** Xiangyu Qi et al. arXiv 2023. [[Paper](https://arxiv.org/pdf/2306.13213.pdf)]

## Alignment Evaluation

### Factuality Evaluation

1. **FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.** Sewon Min et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.14251)]
2. **Factuality Enhanced Language Models for Open-ended Text Generation.** Nayeon Lee et al. NeurIPS 2022. [[Paper](https://arxiv.org/abs/2206.04624)]
3. **TruthfulQA: Measuring How Models Mimic Human Falsehoods.** Stephanie Lin et al. arXiv 2021. [[Paper](https://aclanthology.org/2022.acl-long.229)]
4. **SummaC: Re-visiting NLI-based Models for Inconsistency Detection in Summarization.** Philippe Laban et al. TACL 2022. [[Paper](https://aclanthology.org/2022.tacl-1.10/)]
5. **QAFactEval: Improved QA-based Factual Consistency Evaluation for Summarization.** Alexander R. Fabbri et al. arXiv 2021. [[Paper](https://arxiv.org/abs/2112.08542)]
6. **TRUE: Re-evaluating Factual Consistency Evaluation.** Or Honovich et al. arXiv 2022. [[Paper](https://arxiv.org/abs/2204.04991)]
7. **AlignScore: Evaluating Factual Consistency with a Unified Alignment Function.** Yuheng Zha et al. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16739)]

### Ethics Evaluation

1. **Social Chemistry 101: Learning to Reason about Social and Moral Norms.** Maxwell Forbes et al. arXiv 2020. [[Paper](https://aclanthology.org/2020.emnlp-main.48/)]
2. **Aligning AI with Shared Human Values.** Dan Hendrycks et al. arXiv 2020. [[Paper](https://arxiv.org/abs/2008.02275)]
3. **Would You Rather? A New Benchmark for Learning Machine Alignment with Cultural Values and Social Preferences.** Yi Tay et al. ACL 2020. [[Paper](https://aclanthology.org/2020.acl-main.477/)]
4. **Scruples: A Corpus of Community Ethical Judgments on 32,000 Real-life Anecdotes.** Nicholas Lourie et al. AAAI 2021. [[Paper](https://arxiv.org/abs/2008.09094)]

### Toxicity Evaluation

#### Task-specific Evaluation

1. **Detecting Offensive Language in Social Media to Protect Adolescent Online Safety.** Ying Chen et al. PASSAT-SocialCom 2012. [[Paper](http://www.cse.psu.edu/~sxz16/papers/SocialCom2012.pdf)]
2. **Offensive Language Detection Using Multi-level Classification.** Amir H. Razavi et al. Canadian AI 2010. [[Paper](https://www.cs.csustan.edu/~mmartin/LDS/Razavi.pdf)]
3. **Hateful Symbols or Hateful People? Predictive Features for Hate Speech Detection on Twitter.** Zeerak Waseem and Dirk Hovy. NAACL Student Research Workshop 2016. [[Paper](https://aclanthology.org/N16-2013.pdf)]
4. **Measuring the Reliability of Hate Speech Annotations: The Case of the European Refugee Crisis.** Bjorn Ross et al. NLP4CMC 2016. [[Paper](https://linguistics.rub.de/forschung/arbeitsberichte/17.pdf#page=12)]
5. **Ex Machina: Personal Attacks Seen at Scale.** Ellery Wulczyn et al. WWW 2017. [[Paper](https://arxiv.org/pdf/1610.08914.pdf)]
6. **Predicting the Type and Target of Offensive Posts in Social Media.** Marcos Zampieri et al. NAACL-HLT 2019. [[Paper](https://arxiv.org/pdf/1902.09666)]

#### LLM-centered Evaluation

1. **Recipes for Safety in Open-Domain Chatbots.** Jing Xu et al. arXiv 2020. [[Paper](https://arxiv.org/pdf/2010.07079)]
2. **RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models.** Samuel Gehman et al. EMNLP 2020 Findings. [[Paper](https://arxiv.org/pdf/2009.11462.pdf)]
3. **COLD: A Benchmark for Chinese Offensive Language Detection.** Jiawen Deng et al. EMNLP 2022. [[Paper](https://arxiv.org/pdf/2201.06025)]

### Stereotype and Bias Evaluation

#### Task-specific Evaluation

1. **Gender Bias in Coreference Resolution.** Rachel Rudinger et al. NAACL 2018. [[Paper](https://arxiv.org/pdf/1804.09301)]
2. **Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods.** Jieyu Zhao et al. NAACL 2018. [[Paper](https://arxiv.org/pdf/1804.06876)]
3. **The Winograd Schema Challenge.** Hector Levesque et al. KR 2012. [[Paper](https://cdn.aaai.org/ocs/4492/4492-21843-1-PB.pdf)]
4. **Toward Gender-Inclusive Coreference Resolution: An Analysis of Gender and Bias Throughout the Machine Learning Lifecycle.** Yang Trista Cao and Hal Daumé III. Computational Linguistics 2021. [[Paper](https://aclanthology.org/2021.cl-3.19.pdf)]
5. **Evaluating Gender Bias in Machine Translation.** Gabriel Stanovsky et al. ACL 2019. [[Paper](https://arxiv.org/pdf/1906.00591)]
6. **Investigating Failures of Automatic Translation in the Case of Unambiguous Gender.** Adithya Renduchintala and Adina Williams. ACL 2022. [[Paper](https://arxiv.org/pdf/2104.07838)]
7. **Towards Understanding Gender Bias in Relation Extraction.** Andrew Gaut et al. ACL 2020. [[Paper](https://arxiv.org/pdf/1911.03642)]
8. **Addressing Age-Related Bias in Sentiment Analysis.** Mark Díaz et al. CHI 2018. [[Paper](https://dl.acm.org/doi/pdf/10.1145/3173574.3173986)]
9. **Examining Gender and Race Bias in Two Hundred Sentiment Analysis Systems.** Svetlana Kiritchenko and Saif M. Mohammad. NAACL-HLT 2018. [[Paper](https://arxiv.org/pdf/1805.04508)]
10. **On Measuring and Mitigating Biased Inferences of Word Embeddings.** Sunipa Dev et al. AAAI 2020. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/6267/6123)]
11. **Social Bias Frames: Reasoning About Social and Power Implications of Language.** Maarten Sap et al. ACL 2020. [[Paper](https://arxiv.org/pdf/1911.03891)]
12. **Towards Identifying Social Bias in Dialog Systems: Framework, Dataset, and Benchmark.** Jingyan Zhou et al. EMNLP 2022 Findings. [[Paper](https://arxiv.org/pdf/2202.08011)]
13. **CORGI-PM: A Chinese Corpus for Gender Bias Probing and Mitigation.** Ge Zhang et al. arXiv 2023. [[Paper](https://arxiv.org/pdf/2301.00395)]

#### LLM-centered Evaluation

1. **StereoSet: Measuring Stereotypical Bias in Pretrained Language Models.** Moin Nadeem et al. ACL 2021. [[Paper](https://arxiv.org/pdf/2004.09456)]
2. **CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models.** Nikita Nangia et al. EMNLP 2020. [[Paper](https://arxiv.org/pdf/2010.00133)]
3. **BOLD: Dataset and Metrics for Measuring Biases in Open-Ended Language Generation.** Jwala Dhamala et al. FAccT 2021. [[Paper](https://arxiv.org/abs/2101.11718)]
4. **“I’m sorry to hear that”: Finding New Biases in Language Models with a Holistic Descriptor Dataset.** Eric Michael Smith et al. EMNLP 2022. [[Paper](https://aclanthology.org/2022.emnlp-main.625.pdf)]
5. **Multilingual Holistic Bias: Extending Descriptors and Patterns to Unveil Demographic Biases in Languages at Scale.** Marta R. Costa-jussà et al. arXiv 2023. [[Paper](https://arxiv.org/pdf/2305.13198)]
6. **UNQOVERing Stereotyping Biases via Underspecified Questions.** Tao Li et al. EMNLP 2020 Findings. [[Paper](https://arxiv.org/pdf/2010.02428.pdf)]
7. **BBQ: A Hand-Built Bias Benchmark for Question Answering.** Alicia Parrish et al. ACL 2022 Findings. [[Paper](https://arxiv.org/pdf/2110.08193)]
8. **CBBQ: A Chinese Bias Benchmark Dataset Curated with Human-AI Collaboration for Large Language Models.** Yufei Huang and Deyi Xiong. arXiv 2023. [[Paper](https://arxiv.org/pdf/2306.16244)]

#### Hate Speech Detection

1. **Automated Hate Speech Detection and the Problem of Offensive Language.** Thomas Davidson et al. AAAI 2017. [[Paper](https://www.researchgate.net/profile/Ingmar-Weber/publication/314942659_Automated_Hate_Speech_Detection_and_the_Problem_of_Offensive_Language/links/58e76b6a4585152528de68f2/Automated-Hate-Speech-Detection-and-the-Problem-of-Offensive-Language.pdf)]
2. **Deep Learning for Hate Speech Detection in Tweets.** Pinkesh Badjatiya et al. WWW 2017. [[Paper](https://arxiv.org/pdf/1706.00188)]
3. **Detecting Hate Speech on the World Wide Web.** William Warner and Julia Hirschberg. NAACL-HLT 2012. [[Paper](https://aclanthology.org/W12-2103.pdf)]
4. **A Survey on Hate Speech Detection using Natural Language Processing.** Anna Schmidt and Michael Wiegand. SocialNLP 2017. [[Paper](https://aclanthology.org/W17-1101.pdf)]
5. **Hate Speech Detection with Comment Embeddings.** Nemanja Djuric et al. WWW 2015. [[Paper](https://djurikom.github.io/pdfs/djuric2015wwwB.pdf)]
6. **Are You a Racist or Am I Seeing Things? Annotator Influence on Hate Speech Detection on Twitter.** Zeerak Waseem. NLP+CSS@EMNLP 2016. [[Paper](https://aclanthology.org/W16-5618.pdf)]
7. **TweetBLM: A Hate Speech Dataset and Analysis of Black Lives Matter-related Microblogs on Twitter.** Sumit Kumar and Raj Ratn Pranesh. arXiv 2021. [[Paper](https://arxiv.org/pdf/2108.12521)]
8. **Hate Speech Dataset from a White Supremacy Forum.** Ona de Gibert et al. ALW2 2018. [[Paper](https://arxiv.org/pdf/1809.04444)]
9. **The Gab Hate Corpus: A Collection of 27k Posts Annotated for Hate Speech.** Brendan Kennedy et al. LRE 2022 [[Paper](https://www.researchgate.net/profile/Brendan-Kennedy-4/publication/346608617_The_Gab_Hate_Corpus_A_collection_of_27k_posts_annotated_for_hate_speech/links/5fc932eba6fdcc697bdb7175/The-Gab-Hate-Corpus-A-collection-of-27k-posts-annotated-for-hate-speech.pdf)]
10. **Finding Microaggressions in the Wild: A Case for Locating Elusive Phenomena in Social Media Posts.** Luke Breitfeller et al. EMNLP 2019. [[Paper](https://aclanthology.org/D19-1176.pdf)]
11. **Learning from the Worst: Dynamically Generated Datasets to Improve Online Hate Detection.** Bertie Vidgen et al. ACL 2021. [[Paper](https://arxiv.org/pdf/2012.15761)]
12. **Hate speech detection: Challenges and solutions.** Sean MacAvaney et al. PloS One 2019. [[Paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0221152)]
13. **Racial Microaggressions in Everyday Life: Implications for Clinical Practice.** Derald Wing Sue et al. American Psychologist 2007. [[Paper](https://www.law.stanford.edu/wp-content/uploads/sites/default/files/event/263076/media/slspublic/Sue-RadicalMicroagressionsInEverydayLife.pdf)]
14. **The Impact of Racial Microaggressions on Mental Health: Counseling Implications for Clients of Color.** Kevin L. Nadal et al. Journal of Counseling & Development 2014. [[Paper](https://www.researchgate.net/profile/Kevin-Nadal/publication/262412771_The_Impact_of_Racial_Microaggressions_on_Mental_Health_Counseling_Implications_for_Clients_of_Color/links/606fb0d94585150fe993b16b/The-Impact-of-Racial-Microaggressions-on-Mental-Health-Counseling-Implications-for-Clients-of-Color.pdf)]
15. **A Preliminary Report on the Relationship Between Microaggressions Against Black People and Racism Among White College Students.** Jonathan W. Kanter et al. Race and Social Problems 2017. [[Paper](https://link.springer.com/article/10.1007/s12552-017-9214-0)]
16. **Microaggressions and Traumatic Stress: Theory, Research, and Clinical Treatment.** Kevin L. Nadal. American Psychological Association 2018. [[Paper](https://psycnet.apa.org/record/2017-58590-000?doi=1)]
17. **Arabs as Terrorists: Effects of Stereotypes Within Violent Contexts on Attitudes, Perceptions, and Affect.** Muniba Saleem and Craig A. Anderson. Psychology of Violence 2013. [[Paper](http://www.craiganderson.org/wp-content/uploads/caa/abstracts/2010-2014/13SA.pdf)]
18. **Mean Girls? The Influence of Gender Portrayals in Teen Movies on Emerging Adults' Gender-Based Attitudes and Beliefs.** Elizabeth Behm-Morawitz and Dana E. Mastro. Journalism and Mass Communication Quarterly 2008. [[Paper](https://www.researchgate.net/profile/Elizabeth-Behm-Morawitz/publication/237797930_Mean_Girls_The_Influence_of_Gender_Portrayals_in_Teen_Movies_on_Emerging_Adults'_Gender-Based_Attitudes_and_Beliefs/links/57bdfc6c08ae6f1737689537/Mean-Girls-The-Influence-of-Gender-Portrayals-in-Teen-Movies-on-Emerging-Adults-Gender-Based-Attitudes-and-Beliefs.pdf)]
19. **Exposure to Hate Speech Increases Prejudice Through Desensitization.** Wiktor Soral, Michał Bilewicz, and Mikołaj Winiewski. Aggressive behavior 2018. [[Paper](https://www.academia.edu/download/55409445/4_ab.pdf)]
20. **Latent Hatred: A Benchmark for Understanding Implicit Hate Speech.** Mai ElSherief et al. EMNLP 2021. [[Paper](https://arxiv.org/pdf/2109.05322)]
21. **ToxiGen: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection.** Thomas Hartvigsen et al. ACL 2022. [[Paper](https://arxiv.org/pdf/2203.09509)]
22. **An Empirical Study of Metrics to Measure Representational Harms in Pre-Trained Language Models.** Saghar Hosseini, Hamid Palangi, and Ahmed Hassan Awadallah. arXiv 2023. [[Paper](https://arxiv.org/pdf/2301.09211)]

### General Evaluation

1. **TrustGPT: A Benchmark for Trustworthy and Responsible Large Language Models.** Yue Huang et al. arXiv 2023. [[Paper](https://arxiv.org/pdf/2306.11507.pdf)]
2. **Safety Assessment of Chinese Large Language Models.** Hao Sun et al. arXiv 2023. [[Paper](https://arxiv.org/pdf/2304.10436.pdf)]
3. **FLASK: Fine-grained Language Model Evaluation Based on Alignment Skill Sets.** Seonghyeon Ye et al. arXiv 2023. [[Paper](https://arxiv.org/pdf/2307.10928.pdf)]
4. **Judging LLM-as-a-judge with MT-Bench and Chatbot Arena.** Lianmin Zheng et al. arXiv 2023. [[Paper](https://arxiv.org/pdf/2306.05685.pdf)]
5. **Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models.** Aarohi Srivastava et al. arXiv 2023. [[Paper](https://arxiv.org/pdf/2206.04615.pdf)]
6. **A Critical Evaluation of Evaluations for Long-form Question Answering.** Fangyuan Xu et al. arXiv 2023. [[Paper](https://arxiv.org/pdf/2305.18201.pdf)]
7. **AlpacaEval: An Automatic Evaluator of Instruction-following Models.** Xuechen Li et al. Github 2023. [[Github](https://github.com/tatsu-lab/alpaca_eval)]
8. **AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback.** Yann Dubois et al. Github 2023. [[Paper](https://arxiv.org/pdf/2305.14387.pdf)]
9. **PandaLM: An Automatic Evaluation Benchmark for LLM Instruction Tuning Optimization.** Yidong Wang et al. arXiv 2023. [[Paper](https://arxiv.org/pdf/2306.05087.pdf)]
10. **Large Language Models are not Fair Evaluators.** Peiyi Wang et al. arXiv 2023. [[Paper](https://arxiv.org/pdf/2305.17926.pdf)]
11. **G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment.** Yang Liu et al. arXiv 2023. [[Paper](https://arxiv.org/pdf/2303.16634.pdf)]
12. **Benchmarking Foundation Models with Language-Model-as-an-Examiner.** Yushi Bai et al. arXiv 2023. [[Paper](https://arxiv.org/pdf/2306.04181.pdf)]
13. **PRD: Peer Rank and Discussion Improve Large Language Model based Evaluations.** Ruosen Li et al. arXiv 2023. [[Paper](https://arxiv.org/pdf/2307.02762.pdf)]
14. **SELF-INSTRUCT: Aligning Language Models with Self-Generated Instructions.** Yizhong Wang et al. arXiv 2023. [[Paper](https://arxiv.org/pdf/2212.10560.pdf)]
