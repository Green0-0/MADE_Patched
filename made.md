# MADE: Benchmark Environments for Closed-Loop Materials Discovery

Shreshth A. Malik  $^{12}$  Tiarnan Doherty  $^{2}$  Panagiotis Tigas  $^{2}$  Muhammed Razzak  $^{2}$  Stephen J. Roberts  $^{3}$  Aron Walsh  $^{4}$  Yarin Gal  $^{1}$

# Abstract

Existing benchmarks for computational materials discovery primarily evaluate static predictive tasks or isolated computational sub-tasks. While valuable, these evaluations neglect the inherently iterative and adaptive nature of scientific discovery. We introduce MAterials Discovery Environments (MADE), a novel framework for benchmarking end-to-end autonomous materials discovery pipelines. MADE simulates closed-loop discovery campaigns in which an agent or algorithm proposes, evaluates, and refines candidate materials under a constrained oracle budget, capturing the sequential and resource-limited nature of real discovery workflows. We formalize discovery as a search for thermodynamically stable compounds relative to a given convex hull, and evaluate efficacy and efficiency via comparison to baseline algorithms. The framework is flexible; users can compose discovery agents from interchangeable components such as generative models, filters, and planners, enabling the study of arbitrary workflows ranging from fixed pipelines to fully agentic systems with tool use and adaptive decision making. We demonstrate this by conducting systematic experiments across a family of systems, enabling ablation of components in discovery pipelines, and comparison of how methods scale with system complexity.

# 1. Introduction

Scientific discovery inherently runs in a closed loop. Researchers propose hypotheses, run experiments or simulations, and refine their ideas based on the outcomes (Popper, 2005). Failures can be as informative as successes, and strategies adapt as new evidence emerges. Materials dis

$^{1}$ OATML, Department of Computer Science, University of Oxford  $^{2}$ Diffractive Labs  $^{3}$ Machine Learning Research Group, Department of Engineering Science, University of Oxford  $^{4}$ Thomas Young Centre and Department of Materials, Imperial College London. Correspondence to: Shreshth A. Malik <shreshth@robots.ox.ac.uk>.

Preprint. January 30, 2026.

![img-0.jpeg](img-0.jpeg)
Figure 1. Example acquisition curves for different discovery policies on two quinary inter-metallic systems. The acceleration factor (AF) and enhancement factor (EF) are shown for the LLM orchestrator policy with respect to a random generator baseline policy. Shaded regions are standard error in the mean across 5 episodes.

![img-1.jpeg](img-1.jpeg)

covery is no exception; promising candidates are proposed, evaluated, and iteratively refined, often through cycles of exploration, dead ends, and serendipitous insights.

Yet most computational benchmarks for materials discovery assume a one-way process (Figure 2a). Predictive benchmarks evaluate accuracy of predicting properties such as band gap, energy, and forces using fixed datasets (Dunn et al., 2020; Riebesell et al., 2025; Rubungo et al., 2025; Kirklin et al., 2015). Generative model evaluations on the other hand typically measure metrics such as stability, uniqueness and novelty for one-shot batch generation of candidates (Zeni et al., 2025; Merchant et al., 2023; Betala et al., 2025; Xie et al., 2022). These are valuable assessments of sub-components of pipelines, yet they evaluate models in isolation, divorced from the overall discovery workflow. In practice, researchers combine predictive models, generative models, filters, and heuristics in multi-stage pipelines, with the ultimate objective being the efficient experimental discovery of new materials.

In realistic discovery settings, evaluations such as a high-fidelity simulation or physical experiment can be very costly. Thus the efficiency of discovery algorithms under a limited query budget is important. Methods such as Bayesian optimization and active learning (Settles, 2011; Garnett, 2023) provide principled frameworks for adaptive experimentation and have shown strong performance in molecular and materials applications (Lookman et al., 2019; Rohr et al., 2020). These approaches differ in their modeling and acquisition</shreshth@robots.ox.ac.uk>

MADE: Benchmark Environments for Closed-Loop Materials Discovery

![img-2.jpeg](img-2.jpeg)

![img-3.jpeg](img-3.jpeg)

![img-4.jpeg](img-4.jpeg)
Figure 2. Conceptual overview of the MADE benchmark. a) Existing discovery pipelines and benchmarks follow a static filtering process, moving sequentially from generative models to increasingly expensive evaluation methods, without end-to-end feedback. b) MADE simulates a closed-loop discovery environment where agents iteratively propose candidates, receive oracle feedback (formation energy), and update their strategy. c) Modular, extensible components of the benchmark environments.

![img-5.jpeg](img-5.jpeg)

![img-6.jpeg](img-6.jpeg)

strategies, but are typically developed in settings focused on efficiently optimizing a small number of continuous design variables toward a single global objective. Materials discovery, by contrast, is a multi-minima seeking problem that aims to find diverse stable or metastable compounds within a vast, discrete and constrained chemical space.

Concurrently, LLM-based scientific agents have demonstrated increasing capability at orchestrating multi-step workflows, integrating prior knowledge, and adapting strategies given feedback (Lu et al., 2024; Jia et al., 2024; Guo et al., 2025; Novikov et al., 2025; Abhyankar et al., 2025). While some recent benchmarks begin to probe aspects of scientific reasoning and tool use (Wang et al., 2025; Mirza et al., 2024), there remains limited consensus on how to systematically evaluate agentic systems in open-ended, feedback-driven discovery settings (Song et al., 2025). Materials discovery provides a natural testbed for such systems: candidate proposals can be computationally verified in a closed loop, and there exists a rich ecosystem of computational tools and prior literature. This can enable controlled evaluation of discovery-relevant agentic behaviors such as planning and adaptive decision making.

To address these gaps in evaluations, we introduce a family of MAterials Discovery Environments (MADE) that enable

the evaluation of closed-loop discovery pipelines. In MADE, an agent or algorithm sequentially proposes candidate structures, receives feedback from the environment, and adjusts its strategy to efficiently discover novel thermodynamically stable compounds under a limited query budget. MADE is intentionally modular and composable; users can combine arbitrary planners, generators, filters, and scorers into pipelines, or evaluate fully agentic systems that can utilize environmental feedback. All experiments are specified via configuration files, facilitating reproducible comparisons and systematic ablations across discovery strategies.

Crucially, MADE supports discovery-centric evaluation metrics (Delgado-Licona &amp; Abolhasani, 2023; Adesiji et al., 2025). We quantify how quickly a method discovers new materials with respect to a baseline search strategy, extending the discovery-acceleration paradigm beyond machine learning interatomic potential (MLIP) screening (Riebesell et al., 2025) to the full discovery loop. This enables answering questions such as: what is the performance gain of using a better generative model? Do surrogate model rankings meaningfully accelerate discovery? Are agentic LLM systems more efficient than traditional search algorithms? In summary, our contributions are as follows:

- We introduce MADE, a family of environments for

benchmarking closed-loop computational materials discovery, enabling the first systematic evaluation of full pipelines on open-ended discovery metrics.
- We benchmark strategies ranging from random search with generative models to agentic systems, across multiple system complexities, ablating the contributions of different components in pipelines.
- We find that agentic systems and adaptive search algorithms become more important for discovery efficiency as chemical complexity and search spaces scale, and as surrogate models reduce in efficacy.

## 2 MADE: MAterials Discovery Environment

We first define criteria and motivation for the design of our benchmark. We then introduce MADE , the proposed framework for evaluating materials discovery strategies.

### 2.1 Desiderata for Discovery Benchmarks

We argue that a benchmark for materials discovery should satisfy three general desiderata:

- [leftmargin=*]
- Evaluate closed-loop discovery. It should directly measure how effectively an end-to-end closed-loop system finds new materials to enable ablation of components in the pipeline.
- Reflect realistic search challenges in materials science. It should reflect the discrete, structured but sparse, multi-minima landscape of materials discovery.
- Be general, scalable, and method-agnostic. It should support controlled experiments across chemical systems, search-space sizes, and fidelity levels while remaining implementation-agnostic.

We use these desiderata to motivate MADE’s design while remaining independent of specific pipeline implementation choices. We compare MADE to existing discovery benchmarks against these criteria in Appendix A.

### 2.2 Problem Definition

In MADE, an agent or algorithm interacts with a structured chemical environment by proposing candidate materials, receiving oracle feedback, and adapting its strategy over time. The goal is to uncover new thermodynamically stable compounds efficiently under a constrained oracle budget.

Let $S$ denote the chemical search space, where each candidate $s\in S$ is defined by its chemical composition and crystal structure. We assume access to an oracle $O:S\rightarrow\mathbb{R}$ which returns the predicted formation energy per atom $E_{s}$. Let $B\in\mathbb{N}$ denote the oracle query budget, and define $H_{0}\subset S$ as the initial set of known reference materials.

An agent is defined by its discovery policy $\pi$ that depends on the history of observed (structure, energy) pairs,

$\pi:\{(s_{i},E_{i})\}_{i=1}^{t-1}\rightarrow S.$ (1)

At each iteration $t\leq B$, the agent selects the next candidate structure $s_{t}\sim\pi$, the oracle evaluates its energy, $E_{t}=O(s_{t})$, and the candidate is added to the set of known materials. After updating $H_{t}=H_{t-1}\cup\{s_{t}\}$, the convex hull $\mathrm{CH}(H_{t})$ is recomputed. For each candidate $s\in H_{t}$, we calculate its energy above the convex hull as: $\Delta_{\text{ball}}(s,H_{t})\in\mathbb{R}$. A material is considered thermodynamically stable if its energy lies on or below the convex hull,

$S_{\text{stable},t}=\{s\in H_{t}\mid\Delta_{\text{ball}}(s,H_{t})\leq\epsilon\},$ (2)

where $\epsilon$ is a small stability threshold *(Bartel, 2022)*. Algorithm 1 shows a rollout of one episode in MADE. Pseudocode for the relevant classes is given in Appendix B.

Algorithm 1 MADE episode rollout
0: Chemical search space $S$, initial materials $H_{0}$, policy $\pi$, oracle $O$, budget $B$, threshold $\epsilon$
1: Initialize known materials $H\leftarrow H_{0}$
2: Evaluate energies $E_{s}=O(s)$ for all $s\in H$
3: Construct convex hull $\mathrm{CH}(H)$
4: for $t=1$ to $B$ do
5: $s_{t}\leftarrow\pi(\{(s,E_{s}):s\in H\})$
6: $E_{t}\leftarrow O(s_{t})$
7: $H\leftarrow H\cup\{s_{t}\}$
8: Update $\mathrm{CH}(H)$ and stable set $S_{\text{stable}}$
9: end for
10: Return: $S_{\text{stable}}$

The sequence of proposed materials by the strategy is defined as: $Q_{\pi}=\{s_{1},s_{2},\ldots,s_{B}\}\subset S$. The objective is to design a policy $\pi$ that maximizes the total number of new stable materials discovered after $B$ queries:

$\max_{\pi}|Q_{\pi}\cap S_{\text{stable},B}|.$ (3)

This formulation explicitly treats discovery as multi-minima search which encourages diversity compared to black-box optimization objectives *(Abhyankar et al., 2025)*.

### 2.3 Evaluation Metrics

In this work we assume oracle evaluations dominate the cost of discovery, where the cost of each oracle evaluation greatly exceeds the cost of intermediary computation required to plan and propose the query *(Rainforth et al., 2024)*. This is

often the case as the oracle in real-world use-cases is either expensive DFT calculations or wet-lab experiments. We therefore emphasize discovery-centric metrics that explicitly account for sample efficiency.

#### Independent metrics

These metrics evaluate a single policy without reference to a baseline.

- mSUN *(Merchant et al., 2023; Betala et al., 2025)*. The fraction of *(meta)stable*, *unique* and *novel* materials proposed during an episode. A structure $s$ is counted if it is thermodynamically stable [Eq. (2)], not in $H_{0}$, and not among previously proposed structures. Structural novelty is enforced using pymatgen.StructureMatcher, which applies composition- and geometry-based similarity thresholds to prevent trivial perturbations from being counted as new discoveries.
- Area Under the Discovery Curve (AUDC): Let $D_{\pi}(t)$ denote the cumulative number of mSUN structures discovered by the policy after $t$ oracle queries. The AUDC is defined as $\text{AUDC}=\frac{2}{B^{2}}\int_{0}^{B}D_{\pi}(t)dt$, where we normalize such that the maximum AUDC is 1. This captures both *how many* structures are discovered and *how efficiently* they are found.

#### Relative metrics

To enable fair comparison across chemical systems of varying difficulty, we report metrics relative to a baseline strategy, $\pi_{b}$ (Figure 1). These relative metrics expose which strategies perform better on average within the same operational constraints, enabling principled algorithm selection across diverse discovery stacks.

- Acceleration Factor (AF) *(Rohr et al., 2020)*: For a given number $k$ of discoveries, the acceleration factor is $\text{AF}(k)=t_{\pi_{b}}(k)/t_{\pi}(k)$, where $t_{\pi}(k)$ is the number of oracle queries required by the policy to reach $k$ discoveries. AF quantifies how much more efficient a policy is compared to the baseline.
- Enhancement Factor (EF) *(Rohr et al., 2020)*: For a given number $t$ of queries, the enhancement factor is $\text{EF}(t)=D_{\pi}(t)/D_{\pi_{b}}(t)$, measuring the multiplicative improvement in discoveries over the baseline.

#### Additional and Extensible Metrics

While the above metrics focus on discovery efficiency, MADE is easily extended to measure additional metrics, particularly in multi-objective settings. In experiments in this work, for example, we report composition and structural diversity metrics.

### 2.4 Environment

The environment is defined by the chemical system, initial known materials, and an oracle that evaluates proposals.

#### Chemical Systems and Initial Materials

The chemical space for exploration $S$ is defined by the constituent elements that make up the material. This allows for adjustable difficulty by varying system complexity (e.g. easy: binary metal oxides, medium: ternary inter-metallic compounds, hard: quaternary and beyond), and stoichiometry bounds (maximum number of atoms in the unit cell). Initial known structures ($H_{0}$) can be retrieved from existing datasets *(Horton et al., 2025; Barroso-Luque et al., 2024)*. By varying the size and composition of $H_{0}$, MADE can simulate settings ranging from well-explored chemical spaces to data-scarce regimes. New datasets can readily be incorporated as they emerge *(Siron et al., 2025)*.

#### Oracles

For efficient benchmarking and large-scale experimentation, MADE supports MLIP energy oracles, which offer fast approximate evaluations. Although MLIP evaluations are relatively inexpensive, they provide a realistic setting for studying sequential decision making and strategy adaptation. Crucially, MADE abstracts the oracle interface, allowing substitution with higher-fidelity evaluations such as density functional theory (DFT) calculations or experimental validation for simulation of realistic discovery campaigns.

### 2.5 Example Discovery Policies

The discovery policy defined in Equation 1 is intentionally general, encompassing both classical pipelines (Figure 2a) and agentic systems (Figure 2b). We provide examples here, and defer specific experiments to Section 3.2.

#### Modular Pipelines

Many discovery strategies follow modular pipelines composed of four interchangeable components:

- Planner: Selects compositions to explore, e.g., random, heuristic-, uncertainty- or LLM-based.
- Generator: Proposes candidate structures using methods such as AIRSS *(Pickard and Needs, 2011)*, or generative models *(Xie et al., 2022; Park et al., 2025; Zeni et al., 2025; Antunes et al., 2024)*
- Filter: Drops low-quality candidates from the generator e.g. those that are chemically invalid or redundant, e.g., SMACT *(Davies et al., 2016)*), structural uniqueness via pymatgen.StructureMatcher *(Ong et al., 2013)*, or simple geometric constraints such as minimum interatomic distances.
- Selector: Ranks and selects from generated candidates for selection using e.g., heuristics, surrogate models such as MLIPs *(Batatia et al., 2023; Rhodes et al., 2025)*, or with LLMs.

MADE: Benchmark Environments for Closed-Loop Materials Discovery

Table 1. Results for discovery policies averaged across all system sizes and episodes, at a  $0.1\mathrm{eV}$  stability threshold with a query budget of 50. Higher is better for all columns. The error in the final significant figure(s) is given in brackets as the standard error in the mean. Statistically significant top results are highlighted in bold. Details on metrics and experimental setup are given in Sections 2.3 and 3.

|  Policy |   |   | Discovery Performance |   |   |   | Discovery Diversity  |   |   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  Generator | Planner | Selector | AF | EF | AUDC | mSUN | Mean Comp. L1. | Unique Comps. | Unique SGs  |
|  Random | Random | Random | 1.00(0) | 1.00(0) | 0.115(10) | 0.115(10) | 0.98(4) | 6.3(5) | 1.00(0)  |
|  Random | Diversity | Random | 0.95(10) | 1.11(6) | 0.110(10) | 0.120(10) | 0.97(4) | 6.6(5) | 1.00(0)  |
|  Random | LLM | Random | 1.20(13) | 1.45(11) | 0.123(9) | 0.124(9) | 0.48(3) | 5.5(3) | 1.00(0)  |
|  Chemeleon | Random | Random | 1.72(14) | 1.70(9) | 0.192(15) | 0.192(15) | 0.89(3) | 10.4(7) | 2.74(12)  |
|  Chemeleon | Diversity | Random | 2.10(17) | 1.97(9) | 0.192(15) | 0.207(16) | 0.96(3) | 10.8(8) | 3.69(18)  |
|  Chemeleon | LLM | Random | 3.9(4) | 3.33(19) | 0.273(16) | 0.264(15) | 0.70(2) | 10.4(5) | 6.8(3)  |
|  Chemeleon | - | MLIP | 6.4(5) | 5.3(4) | 0.42(2) | 0.39(2) | 0.84(2) | 19.2(10) | 3.25(14)  |
|  LLM Orch. | - | - | 5.4(5) | 6.0(4) | 0.401(16) | 0.400(15) | 0.71(3) | 10.6(4) | 10.4(3)  |

![img-7.jpeg](img-7.jpeg)
Figure 3. End-to-end materials discovery performance of different policies averaged across all system sizes and episodes, against a random generator baseline. Error bars are one standard error in the mean. See Section 3 for details on experimental setup.

Agentic Systems MADE supports fully agentic systems in which an LLM autonomously orchestrates the discovery loop via tool use, internal state tracking, and multi-step reasoning to choose the next structure to evaluate (Jia et al., 2024; Badrinarayanan et al., 2025; Inizan et al., 2025).

# 3. Experiments

We demonstrate the MADE framework by using it to benchmark end-to-end materials discovery on a variety of policies, comparing the contributions of individual components in deterministic pipelines, and end-to-end agentic systems. We then study how discovery performance scales with chemical system complexity and stability thresholds. Full implementation details and extended results are provided in Appendices B and C.

# 3.1. Benchmark Environments

We evaluate discovery performance across multiple chemical systems and random seeds. For each system, we run 5 independent discovery episodes with an oracle query budget of 50. Unless otherwise stated, results are averaged over 10 randomly sampled systems for each of ternary, quaternary, and quinary inter-metallic chemical spaces (3-5 elements). These spaces are especially relevant given recent interest in high-entropy alloys as a relatively unexplored but potentially fruitful search space for various applications (Nakaya &amp; Furukawa, 2024; Yang et al., 2025).

Chemical System and Initial Materials For each chemical system, we construct  $H_{0}$  using structures retrieved from Materials Project with a maximum of 20 atoms in the unit cell (MP-20) via the API (Horton et al., 2025). We recompute formation energies of structures in  $H_{0}$  using the oracle. A stability threshold of  $0.1\mathrm{eV / atom}$  is used by default, with tighter thresholds (0.01 eV/atom) explored in Section 3.4.

Oracle We use a state-of-the-art MLIP (orb-v3-conservative-inf-omat) (Rhodes et al., 2025) as the formation energy oracle. All structures were relaxed (including unit cell parameters) following the same optimization configuration used in Matbench Discovery (Riebesell et al., 2025).

# 3.2. Discovery Policies and Pipeline Components

As in Section 2.5, we categorize strategies by component to isolate their impact on discovery efficiency and diversity. We note that we do not attempt to exhaustively benchmark all possible strategies and generative models. Instead, we show that our framework enables comparison of the utility of each component in an end-to-end pipeline. More details on specific policy implementations are given in Section B.4.

#### 3.2.1 Planners

##### Random

A composition is selected uniformly at random from the allowed compositional space, without regard to prior evaluations. This provides a non-adaptive baseline.

##### Diversity

To encourage exploration, the diversity planner selects the composition that maximizes the minimum Euclidean distance (in normalized composition space) to previously evaluated compositions and $H_{0}$. This biases sampling toward unexplored regions of composition space.

##### LLM-based Planning

The LLM planner uses an LLM to adaptively select compositions. The planner is prompted with previously explored compositions and oracle feedback, and proposes the next composition to explore, balancing exploitation of promising regions against exploration of new compositional space.

#### 3.2.2 Structure Generators

##### Random

Atoms are placed uniformly at random in fractional coordinates, with lattice parameters sampled from $U(3,15)$ Å, and angles from $U(60,120)^{\circ}$. This generator provides an uninformed structure proposal baseline.

##### Chemeleon

We use Chemeleon *(Park et al., 2025)* trained on MP-20 as an example of a generative model for crystal structure prediction. Chemeleon produces plausible crystal structures similar to its training data distribution, providing a strong prior for generating stable structures.

#### 3.2.3 Selectors

##### Random

A structure is chosen uniformly at random from the generations.

##### MLIP

We use a lower fidelity MLIP, MACE-MP-0-medium *(Batatia et al., 2023)* as a surrogate model for ranking candidates. This enables a similar comparison to see how MLIP rankings speed up discovery. Unlike in the other strategies, we generate a large batch (1024) of structures from across the phase diagram (instead of deciding on a specific composition first) and rank them using the MLIP, mirroring Matbench Discovery.

#### 3.2.4 Agentic LLM Orchestrator

Finally, we evaluate a fully agentic LLM-based discovery policy (LLM Orch.) implemented using a ReAct-style control loop *(Yao et al., 2022)*. At each iteration, the agent conditions on the complete history of evaluated structures and their stability outcomes, together with a summary of previously generated but unevaluated candidate structures stored in an internal buffer. The agent iteratively selects actions from a fixed tool set, including composition selection, conditional structure generation (using Chemeleon or direct structure creation), MLIP-based scoring, and flexible buffer querying. This formulation allows the agent to adapt composition-level exploration and structure-level refinement based on accumulated feedback.

After each oracle evaluation, the buffer and evaluation history are updated and used to inform subsequent decisions. Unlike fixed pipelines, which apply a predetermined sequence of steps, the orchestrator dynamically interleaves generation, scoring, and selection based on the current buffer state and prior evaluations, enabling evaluation of long-horizon, feedback-driven decision making. This serves as a baseline illustration of agentic coordination using large-context reasoning; more expressive agents could for example integrate literature retrieval, materials databases, or additional surrogates.

### 3.3 Results: Discovery against Materials Project

We report averaged metrics over all episodes and system sizes for each discovery policy using a random generator as a baseline in Figure 3 and Table 1.

##### Generative models provide strong priors for efficient discovery

As expected, learned generators such as Chemeleon substantially accelerate discovery relative to random baselines, reflecting a strong inductive bias toward plausible, stable structures.

##### MLIP-based selection significantly accelerates discovery

MLIP-based selection yields the largest single performance gain. The Chemeleon + MLIP pipeline achieves the highest AF among non-agentic methods (AF = 6.4) and the largest AUDC, consistent with prior work demonstrating the effectiveness of surrogate screening in materials discovery.

##### Planning accelerates discovery, even with weak generators

Explicit selection of composition spaces to try provides measurable gains beyond generation alone, including in settings with random structure generation. LLM planning in particular achieves significant gains over random acquisition (AF = 1.2, EF = 1.45). When combined with a strong generator, planning yields substantial additional gains: Chemeleon + LLM planning more than doubles performance relative to Chemeleon alone (AF = 3.9).

##### End-to-end agentic systems compete with optimized modular pipelines

The fully agentic LLM orchestrator achieves discovery efficiency comparable to the strongest modular pipelines, with significantly improved enhancement factor (EF = 6.0) and competitive AUDC and mSUN (Table 1). While its acceleration factor is slightly lower

MADE: Benchmark Environments for Closed-Loop Materials Discovery

![img-8.jpeg](img-8.jpeg)
Figure 4. Performance of policies at increasing system size. Shaded regions are standard error in the mean across 10 systems with 5 episodes each. We see larger gains for effective planning on larger search spaces over baselines.

![img-9.jpeg](img-9.jpeg)
Figure 5. Performance of policies at varying stability threshold for discovery. Shaded regions are standard error in the mean across 10 systems with 5 episodes each. Surrogate model ranking (MLIPs) do not generalise well to smaller tolerances due to errors, whereas planning algorithms lead to significant gains over baselines.

than the best MLIP-ranked pipeline, the orchestrator consistently discovers a broader range of space groups, indicating a different efficiency-diversity trade-off (Section 3.5). This suggests LLMs can effectively plan and optimize for efficient discovery.

# 3.4. Results: Scaling with System Difficulty

Next we examine how discovery performance changes as the search problem becomes more challenging, focusing on system size (Figure 4) and stability threshold (Figure 5).

Planning gains importance as system size increases As the number of constituent elements increases, the number of possible compositions grows combinatorially, making discovery increasingly sparse (Appendix Figure 8). In this regime, adaptive planning yields progressively larger gains over baselines. Figure 4 shows that planning-based strategies, in particular LLM-guided planners and agents, yield progressively larger gains over baselines as system size increases from ternary to quinary systems. Notably, Random + LLM planning outperforms the Chemeleon baseline in larger systems, indicating that composition-level adaptivity can partially compensate for weak generative priors.

Tighter stability thresholds reduce the effectiveness of surrogate ranking Figure 5 shows that at stricter stability thresholds, MLIP-based ranking degrades in performance. This is due to surrogate error near the convex hull (Riebesell et al., 2025), highlighting the importance of research in uncertainty-aware MLIP screening for use within acquisition strategies (Coscia et al., 2025; Busk et al., 2023; Betala et al., 2025). In contrast to MLIP rankings, planning-based strategies retain significant gains over baselines, reflecting greater robustness when discovery targets lie close to stability boundaries. This suggests that adaptive exploration becomes increasingly important as the discovery task be

![img-10.jpeg](img-10.jpeg)

![img-11.jpeg](img-11.jpeg)

comes more selective. In particular, our evidence suggests that LLM-based planners can be less brittle in this regime, likely due to their ability to incorporate broader contextual signals beyond approximate scores from surrogates.

Together, these results highlight that as discovery problems become more challenging, adaptive strategies play an increasingly important role, underscoring the need for benchmarks that evaluate closed-loop discovery behavior.

# 3.5. Results: Diversity of Discovered Materials

Beyond discovery metrics, we also evaluate diversity in the discovered materials using composition- and structure-level metrics (Table 1). In particular, we report the mean pairwise L1 distance between compositions, the number of unique compositions, and the number of unique space-groups (SGs) amongst the discovered mSUN structures.

We find diversity-based planning yields the broadest coverage of composition space, reflected in higher composition distances and expanded phase-diagram coverage (Figures 6 and 7). In contrast, the LLM orchestrator discovers the widest range of SGs, indicating greater structural diversity within explored compositions. These results highlight trade-offs between efficiency and diversity of different strategies.

# 4. Related Work

Computational Materials Benchmarks Benchmarking has played a central role in computational materials science. Foundational datasets such as the Materials Project and OQMD (Kirklin et al., 2015; Dunn et al., 2020) enable large-scale supervised learning for tasks including formation energy, forces, and band-gap prediction. These benchmarks focus on static predictive accuracy on fixed datasets. Matbench Discovery (Riebesell et al., 2025) shifted focus to

MADE: Benchmark Environments for Closed-Loop Materials Discovery

![img-12.jpeg](img-12.jpeg)
Figure 6. Diversity metric distributions for discovered stable structures averaged across system sizes.

wards discovery-oriented evaluation by measuring a model's ability to rank candidates for stability using MLIPs on held-out structures. While this is a step forward, it still remains a screening benchmark: models operate on a fixed candidate pool without closed-loop adaptation. In contrast, MADE evaluates the closed-loop process of deciding what to generate, what to filter, and what to evaluate next. Meanwhile, papers on generative modeling (Zeni et al., 2025; Merchant et al., 2023; Park et al., 2025; Xie et al., 2022) and recent benchmarks for these models (Betala et al., 2025) focus on evaluating average unconditional generation quality rather than discovery acceleration metrics.

Agentic Systems and AI-Driven Scientific Workflows Recent advances in agentic systems, including LLM-based scientific agents (Lu et al., 2024; Guo et al., 2025; Novikov et al., 2025), tool-using design assistants (Jia et al., 2024; Inizan et al., 2025), and specialized agents for materials workflows (Badrinarayanan et al., 2025; Rubungo et al., 2025), have highlighted the potential of systems capable of multi-step reasoning, tool orchestration, and iterative refinement. Related work has also explored LLMs within classical adaptive search paradigms, such as Bayesian optimization and bandit-style decision making (Liu et al., 2024; Nie et al., 2024). However, most existing benchmarks assess static reasoning or tool use (Wang et al., 2025; Mirza et al., 2024; Jimenez et al., 2023; Nathani et al., 2025; Zhang et al., 2025) rather than long-horizon, feedback-driven discovery. Concurrent work on science-oriented benchmarks (Song et al., 2025; Huang et al., 2025) have started to move toward hypothesis-experiment-observation loops, but re

![img-13.jpeg](img-13.jpeg)
Figure 7. Starting (MP-20) and post-acquisition phase diagrams under different strategies on an example ternary system.

main focused on structured evaluation settings rather than end-to-end discovery, and while Abhyankar et al. (2025) apply LLMs directly to materials discovery, evaluation only compares to generative models rather than full pipelines.

Active learning and Bayesian Optimization Active learning, Bayesian optimization (BO), and related experimental design methods provide a principled framework for sequential decision making under uncertainty, enabling efficiency optimization under limited evaluation budgets (Settles, 2011; Garnett, 2023; Rainforth et al., 2024). These methods combine surrogate models with acquisition functions that balance exploration and exploitation, and have been widely applied in materials science to optimize properties in low-dimensional design spaces such as mapping phase diagrams (Lookman et al., 2019; Kusne et al., 2020; Rohr et al., 2020; Wang et al., 2022; Novick et al., 2024). Unlike classical black-box optimization, which targets a single global optimum, materials discovery is inherently multimodal, seeking a diverse set of local minima corresponding to stable or metastable compounds for experimental verification. MADE enables integration and evaluation of BO strategies within broader discovery pipelines.

# 5. Discussion and Conclusions

We introduce MADE, a family of benchmark environments that reframe materials discovery as a closed-loop sequential decision making task, enabling flexible and scalable evaluation of end-to-end discovery beyond what is possible with existing static benchmarks. Using MADE, we show that while pipelines reliant on surrogate model screening perform well for simple systems, adaptive planning strategies become increasingly important as search spaces grow and as surrogate errors become more consequential or out-of

MADE: Benchmark Environments for Closed-Loop Materials Discovery

distribution.

#### Limitations and Future Work

Current experiments rely on generators and MLIPs trained on MP-20, and thus inherit shared distributional biases that may simplify discovery relative to real-world settings. Extending MADE to incorporate DFT or experimental oracles, batched query evaluation, and multi-objective tasks are natural next steps. As a gym-like environment, MADE also enables reinforcement learning over the full discovery loop, connecting to recent work on fine-tuning crystal generators and adaptive policies *(Chen et al., 2025; Park and Walsh, 2025)*.

#### Outlook

More broadly, MADE provides a concrete testbed for evaluating core capabilities of agentic systems in realistic scientific discovery settings, including long-horizon planning, reasoning under uncertainty, and learning from feedback. By enabling controlled evaluation of these behaviors in closed-loop environments, MADE can help ground progress toward autonomous scientific discovery systems.

## Acknowledgments

The authors would like to thank Atılım Güneş Baydin for useful feedback on the paper. SM acknowledges funding from the EPSRC Centre for Doctoral Training in Autonomous Intelligent Machines and Systems (Grant No: EP/S024050/1). YG acknowledges funding from the Turing Fellowship (Grant No. EP/V030302/1). We acknowledge funding from the Modal academic grant program for compute credits.

## Impact Statement

As aspects of scientific research become increasingly automated, there is a growing need to critically assess the benefits and risks of handing over research autonomy to AI systems. Benchmark frameworks such as the proposed can help surface and study the risks by making agent behavior and decision making processes evident on test-beds before deployment. More broadly, defining effective evaluation metrics for autonomous discovery agents is critical to ensuring agents pursue human aligned scientific goals.

## References

- Abhyankar et al. (2024) Abhyankar, N., Kabra, S., Desai, S., and Reddy, C. K. Accelerating materials design via LLM-guided evolutionary search. *arXiv preprint arXiv:2510.22503*, 2025.
- Adesiji et al. (2025) Adesiji, A. D., Wang, J., Kuo, C.-S., and Brown, K. A. Benchmarking self-driving labs. *arXiv preprint arXiv:2508.06642*, 2025.
- Antunes et al. (2024) Antunes, L. M., Butler, K. T., and Grau-Crespo, R. Crystal structure generation with autoregressive large language modeling. *Nature Communications*, 15(1):10570, 2024.
- Badrinarayanan et al. (2024) Badrinarayanan, S., Magar, R., Antony, A., Meda, R. S., and Farimani, A. B. MOFGPT: Generative design of metal-organic frameworks using language models. *arXiv preprint arXiv:2506.00198*, 2025.
- Barroso-Luque et al. (2024) Barroso-Luque, L., Shuaibi, M., Fu, X., Wood, B. M., Dzamba, M., Gao, M., Rizvi, A., Zitnick, C. L., and Ulissi, Z. W. Open materials 2024 (OMat24) inorganic materials dataset and models. *arXiv preprint arXiv:2410.12771*, 2024.
- Bartel et al. (2023) Bartel, C. J. Review of computational approaches to predict the thermodynamic stability of inorganic solids. *Journal of Materials Science*, 57(23):10475–10498, 2022.
- Batatia et al. (2023) Batatia, I., Benner, P., Chiang, Y., Elena, A. M., Kovács, D. P., Riebesell, J., Advincula, X. R., Asta, M., Avaylon, M., Baldwin, W. J., et al. A foundation model for atomistic materials chemistry. *arXiv preprint arXiv:2401.00096*, 2023.
- Betala et al. (2023) Betala, S., Gleason, S. P., Ramlaoui, A., Xu, A., Channing, G., Levy, D., Fourrier, C., Kazeev, N., Joshi, C. K., Kaba, S.-O., et al. LeMat-GenBench: A unified evaluation framework for crystal generative models. *arXiv preprint arXiv:2512.04562*, 2025.
- Bitzek et al. (2023) Bitzek, E., Koskinen, P., Gähler, F., Moseler, M., and Gumbsch, P. Structural relaxation made simple. *Physical review letters*, 97(17):170201, 2006.
- Busk et al. (2023) Busk, J., Schmidt, M. N., Winther, O., Vegge, T., and Jørgensen, P. B. Graph neural network interatomic potential ensembles with calibrated aleatoric and epistemic uncertainty on energy and forces. *Physical Chemistry Chemical Physics*, 25(37):25828–25837, 2023.
- Chen et al. (2024) Chen, J., Guo, J., Fako, E., and Schwaller, P. Accelerating inverse materials design using generative diffusion models with reinforcement learning. *arXiv preprint arXiv:2511.03112*, 2025.
- Coscia et al. (2023) Coscia, D., de Haan, P., and Welling, M. BLIPs: Bayesian learned interatomic potentials. *arXiv preprint arXiv:2508.14022*, 2025.
- Davies et al. (2023) Davies, D. W., Butler, K. T., Jackson, A. J., Morris, A., Frost, J. M., Skelton, J. M., and Walsh, A. Computational screening of all stoichiometric inorganic materials. *Chem*, 1(4):617–627, 2016.
- Delgado-Licona et al. (2023) Delgado-Licona, F. and Abolhasani, M. Research acceleration in self-driving labs: Technological roadmap toward accelerated materials and molecular discovery. *Advanced Intelligent Systems*, 5(4):2200331, 2023.

MADE: Benchmark Environments for Closed-Loop Materials Discovery

Dunn, A., Wang, Q., Ganose, A., Dopp, D., and Jain, A. Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm. npj Computational Materials, 6(1):138, 2020.
Garnett, R. Bayesian Optimization. Cambridge University Press, 2023.
Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
Horton, M. K., Huck, P., Yang, R. X., Munro, J. M., Dwaraknath, S., Ganose, A. M., Kingsbury, R. S., Wen, M., Shen, J. X., Mathis, T. S., et al. Accelerated data-driven materials science with the Materials Project. Nature Materials, pp. 1-11, 2025.
Huang, X., Chen, J., Fei, Y., Li, Z., Schwaller, P., and Ceder, G. Cascade: Cumulative agentic skill creation through autonomous development and evolution. arXiv preprint arXiv:2512.23880, 2025.
Inizan, T. J., Yang, S., Kaplan, A., Lin, Y.-h., Yin, J., Mirzaei, S., Abdelgaid, M., Alawadhi, A. H., Cho, K., Zheng, Z., et al. System of agentic AI for the discovery of metal-organic frameworks. arXiv preprint arXiv:2504.14110, 2025.
Jia, S., Zhang, C., and Fung, V. LLMatDesign: Autonomous materials discovery with large language models. arXiv preprint arXiv:2406.13163, 2024.
Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., and Narasimhan, K. SWE-Bench: Can language models resolve real-world github issues? arXiv preprint arXiv:2310.06770, 2023.
Khattab, O., Singhvi, A., Maheshwari, P., Zhang, Z., Santhanam, K., Vardhamanan, S., Haq, S., Sharma, A., Joshi, T. T., Moazam, H., et al. DSPy: Compiling declarative language model calls into self-improving pipelines. arXiv preprint arXiv:2310.03714, 2023.
Kirklin, S., Saal, J. E., Meredig, B., Thompson, A., Doak, J. W., Aykol, M., Ruhl, S., and Wolverton, C. The open quantum materials database (OQMD): assessing the accuracy of DFT formation energies. npj Computational Materials, 1(1):1-15, 2015.
Kusne, A. G., Yu, H., Wu, C., Zhang, H., Hattrick-Simpers, J., DeCost, B., Sarker, S., Oses, C., Toher, C., Curtarolo, S., et al. On-the-fly closed-loop materials discovery via bayesian active learning. Nature communications, 11(1): 5966, 2020.

Larsen, A. H., Mortensen, J. J., Blomqvist, J., Castelli, I. E., Christensen, R., Dulak, M., Friis, J., Groves, M. N., Hammer, B., Hargus, C., et al. The atomic simulation environment—a Python library for working with atoms. Journal of Physics: Condensed Matter, 29(27):273002, 2017.
Liu, T., Astorga, N., Seedat, N., and van der Schaar, M. Large language models to enhance Bayesian optimization. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=00xotBmGol.
Lookman, T., Balachandran, P. V., Xue, D., and Yuan, R. Active learning in materials science with emphasis on adaptive sampling using uncertainties for targeted design. npj Computational Materials, 5(1):21, 2019.
Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., and Ha, D. The AI Scientist: Towards fully automated open-ended scientific discovery. arXiv preprint arXiv:2408.06292, 2024.
Merchant, A., Batzner, S., Schoenholz, S. S., Aykol, M., Cheon, G., and Cubuk, E. D. Scaling deep learning for materials discovery. Nature, 624(7990):80-85, 2023.
Mirza, A., Alampara, N., Kunchapu, S., Ríos-García, M., Emoekabu, B., Krishnan, A., Gupta, T., Schilling-Wilhelmi, M., Okereke, M., Aneesh, A., et al. Are large language models superhuman chemists? arXiv preprint arXiv:2404.01475, 2024.
Nakaya, Y. and Furukawa, S. High-entropy intermediaclcs: emerging inorganic materials for designing high-performance catalysts. Chemical Science, 2024.
Nathani, D., Madaan, L., Roberts, N., Bashlykov, N., Menon, A., Moens, V., Budhiraja, A., Magka, D., Vorotilov, V., Chaurasia, G., et al. MLGym: A new framework and benchmark for advancing ai research agents. arXiv preprint arXiv:2502.14499, 2025.
Nie, A., Su, Y., Chang, B., Lee, J. N., Chi, E. H., Le, Q. V., and Chen, M. EVOLvE: Evaluating and optimizing LLMs for exploration. arXiv preprint arXiv:2410.06238, 2024.
Novick, A., Cai, D., Nguyen, Q., Garnett, R., Adams, R., and Toberer, E. Probabilistic prediction of material stability: integrating convex hulls into active learning. *Materials Horizons*, 11(21):5381-5393, 2024.
Novikov, A., Vū, N., Eisenberger, M., Dupont, E., Huang, P.-S., Wagner, A. Z., Shirobokov, S., Kozlovskii, B., Ruiz, F. J., Mehrabian, A., et al. Alphaevolve: A coding agent for scientific and algorithmic discovery. arXiv preprint arXiv:2506.13131, 2025.

MADE: Benchmark Environments for Closed-Loop Materials Discovery

Ong, S. P., Richards, W. D., Jain, A., Hautier, G., Kocher, M., Cholia, S., Gunter, D., Chevrier, V. L., Persson, K. A., and Ceder, G. Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis. Computational Materials Science, 68:314-319, 2013.
Park, H. and Walsh, A. Guiding generative models to uncover diverse and novel crystals via reinforcement learning. arXiv preprint arXiv:2511.07158, 2025.
Park, H., Onwuli, A., and Walsh, A. Exploration of crystal chemical space using text-guided generative artificial intelligence. Nature Communications, 16(1):4379, 2025.
Pickard, C. J. and Needs, R. Ab initio random structure searching. Journal of Physics: Condensed Matter, 23(5): 053201, 2011.
Popper, K. The logic of scientific discovery. Routledge, 2005.
Rainforth, T., Foster, A., Ivanova, D. R., and Bickford Smith, F. Modern bayesian experimental design. Statistical Science, 39(1):100-114, 2024.
Rhodes, B., Vandenhaute, S., Simkus, V., Gin, J., Godwin, J., Duignan, T., and Neumann, M. Orb-v3: atomistic simulation at scale. arXiv preprint arXiv:2504.06231, 2025.
Riebesell, J., Goodall, R. E., Benner, P., Chiang, Y., Deng, B., Ceder, G., Asta, M., Lee, A. A., Jain, A., and Persson, K. A. A framework to evaluate machine learning crystal stability predictions. Nature Machine Intelligence, 7(6): 836-847, 2025.
Rohr, B., Stein, H. S., Guevarra, D., Wang, Y., Haber, J. A., Aykol, M., Suram, S. K., and Gregoire, J. M. Benchmarking the acceleration of materials discovery by sequential learning. Chemical science, 11(10):2696-2706, 2020.
Rubungo, A. N., Li, K., Hattrick-Simpers, J., and Dieng, A. B. LLM4Mat-Bench: benchmarking large language models for materials property prediction. Machine Learning: Science and Technology, 6(2):020501, 2025.
Settles, B. From theories to queries: Active learning in practice. In Active learning and experimental design workshop in conjunction with AISTATS 2010, pp. 1-18. JMLR Workshop and Conference Proceedings, 2011.
Siron, M., Djafar, I., Ramlaoui, A., Fayette, E. d., Rossello, A., Fako, E., McDermott, M., Therrien, F., Barroso-Luque, L., Cipcigan, F., et al. Lemat-bulk: aggregating, and de-duplicating quantum chemistry materials databases. arXiv preprint arXiv:2511.05178, 2025.

Song, Z., Lu, J., Du, Y., Yu, B., Pruyn, T. M., Huang, Y., Guo, K., Luo, X., Qu, Y., Qu, Y., et al. Evaluating large language models in scientific discovery. arXiv preprint arXiv:2512.15567, 2025.
Wang, A., Liang, H., McDannald, A., Takeuchi, I., and Kusne, A. G. Benchmarking active learning strategies for materials optimization and discovery. Oxford Open Materials Science, 2(1):itac006, 2022.
Wang, Z., Huang, H., Zhao, H., Xu, C., Zhu, S., Janssen, J., and Viswanathan, V. DREAMS: Density functional theory based research engine for agentic materials simulation. arXiv preprint arXiv:2507.14267, 2025.
Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824-24837, 2022.
Widdowson, D. and Kurlin, V. Resolving the data ambiguity for periodic crystals. Advances in Neural Information Processing Systems, 35:24625-24638, 2022.
Widdowson, D., Mosca, M. M., Pulido, A., Kurlin, V., and Cooper, A. I. Average minimum distances of periodic point sets - foundational invariants for mapping periodic crystals. MATCH Communications in Mathematical and in Computer Chemistry, 87(3):529-559, 2022. doi: 10.46793/match.87-3.529W.
Xie, T., Fu, X., Ganea, O., Barzilay, R., and Jaakkola, T. S. Crystal diffusion variational autoencoder for periodic material generation. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview.net/forum?id=03RLpj-tc_.
Yadan, O. Hydra - a framework for elegantly configuring complex applications. Github, 2019. URL https://github.com/facebookresearch/hydra.
Yang, Y.-F., Hu, F., Xia, T., Li, R.-H., Bai, J.-Y., Zhu, J.-Q., Xu, J.-Y., and Zhang, G.-F. High entropy alloys: A review of preparation techniques, properties and industry applications. Journal of Alloys and Compounds, 1010: 177691, 2025.
Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. R., and Cao, Y. ReAct: Synergizing reasoning and acting in language models. In The eleventh international conference on learning representations, 2022.
Zeni, C., Pinsler, R., Zügner, D., Fowler, A., Horton, M., Fu, X., Wang, Z., Shysheya, A., Crabbé, J., Ueda, S., et al. A generative model for inorganic materials design. Nature, 639(8055):624-632, 2025.

MADE: Benchmark Environments for Closed-Loop Materials Discovery

Zhang, J., Gan, J., Wang, X., Jia, Z., Gu, C., Chen, J., Zhu, Y., Ma, M. D., Zhou, D., Li, L., et al. MatSciBench: Benchmarking the reasoning ability of large language models in materials science. arXiv preprint arXiv:2510.12171, 2025.

MADE: Benchmark Environments for Closed-Loop Materials Discovery

# A. Comparison of Materials Discovery Benchmarks

Table 2 provides a high-level comparison of materials discovery benchmarks under the criteria outlined in Section 2.1.

MLIP screening benchmarks (e.g. Riebesell et al. (2025)) that rank candidates in a pool are method-agnostic but are not open-ended; they require a fixed candidate pool set, nor do they incorporate closed-loop feedback.

Generative model benchmarks (e.g. Betala et al. (2025)) enable open-ended generative evaluation but do not incorporate closed-loop feedback or discovery acceleration of end-to-end pipelines.

Prior materials optimization benchmarks (e.g. Abhyankar et al. (2025)) do incorporate feedback but do not provide standardized environments for comparing methods, and often only evaluate against static generative model baselines rather than the model in the context of a discovery pipeline, and only consider top performance of an objective rather than discovery acceleration metrics.

MADE uniquely evaluates discovery acceleration metrics for end-to-end, iterative, and open-ended tasks with oracle feedback, while remaining agnostic to the choice of search or modeling strategy.

Table 2. Comparison of MADE with existing classes of materials discovery benchmarks. Legend:  $\checkmark$  supported,  $\sim$  partially supported,  $\times$  not supported

|  Benchmark Class | End-to-End | Closed-Loop | Open-Ended | Discovery Metrics | Method-Agnostic  |
| --- | --- | --- | --- | --- | --- |
|  MLIP screening benchmarks | × | × | × | ~ | ✓  |
|  Generative model benchmarks | × | × | ✓ | × | ✓  |
|  Materials optimization benchmarks | ~ | ✓ | ✓ | ~ | ~  |
|  MADE (ours) | ✓ | ✓ | ✓ | ✓ | ✓  |

# B. Implementation Details

# B.1. General Computational Details

Configuration and Reproducibility Pipelines, components, and environments are all instantiated through declarative YAML configuration files using Hydra (Yadan, 2019), enabling controlled comparison of strategies by changing a single module while holding others fixed. This design promotes reproducibility by making all experimental choices (models, budgets, seeds, thresholds etc.) explicit and versionable, and lowers the barrier to implementing and evaluating new components within the same experimental protocol. All code, configurations, and scripts used to generate the results in this paper are publicly released in the code repository for reproducibility.

Compute Experiments were run on a single NVIDIA T4 GPU per episode. Episodes were parallelized across systems and random seeds using Modal cloud compute.

LLM Usage All LLM-based components use a single reasoning-capable language model (GPT-5.1 via the OpenAI API). We use DSPy (Khattab et al., 2023) as the backend for structured prompt design and outputs. In DSPy, prompts are defined in through inputs and output typed signatures rather than free-form text prompts. Each signature specifies named input fields (e.g., evaluation history) and output fields, which DSPy automatically serializes into a consistent prompt format for the language model. This approach enables reproducible prompt construction, modular composition of reasoning components (such as chain-of-thought (Wei et al., 2022) or ReAct (Yao et al., 2022)), and systematic control over what information is exposed to the model at each step.

# B.2. MADE Pseudocode

Listing 1 contains pseudocode for the base classes of the MADE benchmark. The Oracle, Environment and Agent base classes are flexible to allow for different methods to be implemented. We make use of pymatgen (Ong et al., 2013) classes to use phase diagrams as the environment state and convex hull computations.

Listing 1. Pseudocode for key classes in MADE

```python
class Oracle: def __init__(self, model):

self.model = model # e.g., MLIP, DFT, experimental oracle

def predict_energy(self, structure):
energy = self.model.predict(structure)
return energy

class Environment:
def __init__(self, oracle, initial_known_structures, chemical_system):
self.oracle = oracle
self.chemical_system = chemical_system
self.known_structures = initial_known_structures
self.energies = {s: self.oracle.predict_energy(s)
for s in initial_known_structures}
self.update_convex_hull()

def step(self, structure):
energy = self.oracle.predict_energy(structure)
self.known_structures.append(structure)
self.energies[structure] = energy
self.update_convex_hull()
return energy

def update_convex_hull(self):
self.convex_hull = ConvexHull(self.known_structures, self.energies)

def reset(self):
self.known_structures.clear()
self.energies.clear()
self.update_convex_hull()

class Agent:
def __init__(self, chemical_system):
self.chemical_system = chemical_system

self.policy = Policy(chemical_system)

def predict_next_structure(self, env):
# policy can use existing convex hull and query history to propose next structure
next_structure = self.policy(env)
return next_structure

# example rollout
oracle = Oracle(model)
env = Environment(oracle, initial_known_structures, chemical_system)
agent = Agent(chemical_system)

for t in range(query_budget):
structure = agent.predict_next_structure(env)
energy = env.step(structure)

### B.3 Benchmark Environments

#### Chemical Systems

As described in Section 3.1, we evaluate discovery across inter-metallic chemical systems of increasing complexity. We randomly sample 10 systems from each of ternary, quaternary, and quinary element spaces (3–5 elements), excluding systems with radioactive elements. The systems used are shown in Table 3. Each system is evaluated over 5 independent discovery episodes.

#### Oracle

As mentioned in Section 3.1, We use orb-v3-conservative-inf-omat *(Rhodes et al., 2025)* as the formation energy oracle. All structures were relaxed (including unit cell parameters) using the FIRE optimizer *(Bitzek et al.,

MADE: Benchmark Environments for Closed-Loop Materials Discovery

Table 3. Chemical systems used for benchmarking, grouped by system size.

|  Ternary systems | Quaternary systems | Quinary systems  |
| --- | --- | --- |
|  Mg-Sn-Sr | Ba-Nd-Ni-W | Cd-Li-Nd-Ti-W  |
|  Au-K-Tb | Eu-Nb-Sn-Tl | Cd-Gd-Mn-Na-Ta  |
|  Co-Mg-Na | Ce-Er-Pb-Rh | Co-Hg-Mg-Sr-W  |
|  Hf-Ni-Zr | Dy-K-Pd-Sm | Ca-Fe-Gd-Pb-Tb  |
|  Co-Dy-W | Co-Dy-Ta-Y | Al-Hg-K-Mg-W  |
|  Ga-Pt-Tm | Au-Cr-Cs-Dy | Ag-Nd-Pd-Pt-Tb  |
|  Ga-Ho-Lu | Ca-Pd-Sn-W | Al-Lu-Pt-Rb-Sm  |
|  Al-Li-V | Au-Tb-V-Y | Co-Hf-In-Ru-Tm  |
|  Al-V-Zn | Ce-Ir-Pt-Sn | Ho-In-Mg-Pd-Zr  |
|  Co-Pd-Tl | Ba-Be-Hf-Li | Cr-Fe-Lu-Pt-Sc  |

2006) for a maximum of 500 steps or fmax of 0.02 in ASE (Larsen et al., 2017) before evaluating the final energy, mirroring Matbench Discovery (Riebesell et al., 2025).

Structure Matching Structural uniqueness is computed using pymatgen. StructureMatcher on the primitive cell of each structure with the default lattice, site, and angle tolerances (1tol=0.2, stol=0.3, angle_tol=5.0). Further work could migrate to make use of recent efforts improving structure matching (Siron et al., 2025).

Note on Discovery Metrics If, for a given system, the baseline policy does not find any new structures, the acceleration factor is not defined. In these settings, we define the acceleration factor to be equal to the maximum number of queries (50). This was the case for a small number of quinary systems.

# B.4. Specific Policy Details

General Workflow for Non-Agentic Policies Unless otherwise stated, each oracle query proceeds by first selecting a single composition using the planner, then generating a batch of 32 candidate structures conditioned on that composition using the generator, from which a single structure is selected for evaluation using the selector.

# B.4.1. RANDOM GENERATOR

The random generator generates crystal structures by randomly assigning lattice parameters  $a, b, c$  from the uniform distribution  $U(3,15)$  Å, and angles  $\alpha, \beta, \gamma$  from  $U(60,120)$  degrees. and fractional atomic positions from  $U(0,1)$ . While more sophisticated heuristics could adapt lattice constants to unit-cell size, we use this simple formulation to provide a minimal baseline.

# B.4.2. CHEMELEON GENERATOR

We use Chemeleon (Park et al., 2025) trained on MP-20 to produce structure proposals conditioned on composition. For planner-based policies, we generate 32 candidate structures for the selected composition at each query step. For the MLIP-ranked policy, we generate 1024 structures across randomly sampled valid compositions to mimic common high-throughput generative screening workflows. This roughly matches the same total number of generations over the episode as the sequential planning setting  $(50 \times 32)$  for fair comparison.

# B.4.3. CHEMELEON + MLIP RANKING

For MLIP-based baselines, we generate a large batch (1024) of candidate structures across the phase diagram and rank all candidates using a lower-fidelity MLIP (MACE MP-0-medium (Batatia et al., 2023)) surrogate before oracle evaluation. This mirrors common MatBench-style discovery pipelines and isolates the effect of surrogate-based ranking without adaptive planning.

# B.4.4. DIVERSITY PLANNER

The diversity planner selects compositions to maximize coverage of composition space while accounting for prior exploration outcomes. All compositions up to a maximum stoichiometry are enumerated at initialization. Each composition is represented

as a vector of fractional elemental concentrations, and pairwise distances (Euclidean by default) are computed between candidate compositions and a reference set consisting of previously attempted compositions and elemental end members.

Each composition $c$ is represented by a normalized composition vector

$\mathbf{x}_{c}=\big{(}x_{c,1},x_{c,2},\ldots,x_{c,d}\big{)},$

where $x_{c,i}$ denotes the fractional concentration of element $i$ and $d$ is the number of elements in the chemical system. Let $\mathcal{R}$ denote the reference set of compositions, consisting of all previously attempted compositions together with elemental end members. We then define the valid reference set for a composition as,

$\mathcal{R}_{c}=\{c^{\prime}\in\mathcal{R}\mid\text{red}(c^{\prime})\neq\text{red}(c)\}\,,$

where $\mathrm{red}(\cdot)$ denotes the reduced chemical formula. This mask helps avoid the minimum distances being trivially zero for the same reduced compositions. The diversity distance for composition $c$ is then defined as the minimum Euclidean distance to this masked reference set:

$D(c)=\min_{c^{\prime}\in\mathcal{R}_{c}}\left\|\mathbf{x}_{c}-\mathbf{x}_{c^{\prime}}\right\|_{2}=\min_{c^{\prime}\in\mathcal{R}_{c}}\sqrt{\sum_{i=1}^{d}\left(x_{c,i}-x_{c^{\prime},i}\right)^{2}}.$ (4)

The diversity score is multiplied by a composition-specific weight that encodes exploration history. Unattempted compositions receive a fixed weight (5.0), strongly prioritizing unexplored regions of composition space. For previously attempted compositions, the weight is computed as

$w(c)=\alpha\cdot\frac{1}{n_{c}+1}+\beta\cdot(1-r_{c}),$ (5)

where $n_{c}$ is the number of attempts for composition $c$, $r_{c}$ is its empirical success rate, and $(\alpha,\beta)=(0.7,0.3)$. The planner selects the composition with the highest weighted diversity scores $w(c)D(c)$, encouraging systematic exploration of sparse regions while still revisiting compositions with unresolved failures if all compositions have been attempted.

#### B.4.5 LLM Planner

The LLM planner operates at the composition level, selecting which compositions to explore based on accumulated oracle feedback. At each planning step, the raw environment state is summarized into a structured context dictionary, which is then passed to a DSPy signature and automatically serialized into a prompt.

Concretely, the planner’s input fields include: (i) the allowed chemical elements defining the search space, (ii) the stability threshold and query count, (iii) a summarized list of previously evaluated entries sorted by energy above the convex hull (including reduced and full formulas, energies, and stability labels), (iv) composition-level trial counts for both reduced formulas (phase diagram points) and full formulas (unit cell sizes), and (v) the most recent query and observation. To control context length, the number of summarized entries can be capped, with stable or metastable entries always included and the remainder randomly sampled. We place the cap at 20 structures. The planner produces a single structured output: a list of candidate compositions specified as full formulas with explicit unit cell sizes, which are subsequently validated against element and stoichiometry constraints before structure generation. The system prompt used for the LLM planner is provided in Listing 2.

Listing 2: LLM Planner Prompt
⬇
You are a planner for a material discovery experiment. Your goal is to discover as many NOVEL, UNIQUE, STABLE (or metastable) structures as possible.

CRITICAL CONSTRAINT: You MUST ONLY use elements from the provided ’elements’ list in your compositions. And you MUST ONLY propose compositions that are within the max_stoichiometry.
- Example: If elements=[’Li’, ’O’], you can propose Li2O, LiO2, etc., but NOT Na2O, Fe2O3, etc.
- Example: If max_stoichiometry=20, you can propose Li2O, LiO2, etc., but NOT Li19O19.

DEFINITIONS:– STABLE/METASTABLE: Structures with e_above_hull <= stability_tolerance.– NOVEL: Not already known on the convex hull (is_newly_discovered=True).– UNIQUE: Structurally distinct from previously evaluated structures.– Entries marked is_stable_or_metastable=True with is_newly_discovered=True are successful discoveries.

PHASE DIAGRAM CONCEPTS:– Reduced formulas (e.g., Li20) represent UNIQUE POINTS on the phase diagram.– Different reduced compositions = different phase diagram points, then PRIORITIZE DIVERSE reduced compositions for broad coverage.

STRUCTURE DIVERSITY AT SAME COMPOSITION:– Multiple DIFFERENT structures can exist at the SAME reduced composition (same phase diagram point)– Different structures for the SAME composition can have DIFFERENT stabilities (one unstable doesn’t mean all are!)– Different unit cell sizes (Li20 vs Li402) create different structures but occupy the SAME phase diagram point

STRATEGY:– Explore diverse reduced compositions (different phase diagram points) to maximize phase diagram coverage.– If a reduced composition has yielded stable/metastable NOVEL structures, consider proposing MORE unit cell sizes for it as additional stable polymorphs may exist.– Compositions with only [unstable] entries may still have stable structures.– Balance exploration (new reduced compositions) vs exploitation (trying to find stable structures for compositions with only [unstable] entries)

Propose FULL formulas with specific unit cell sizes (e.g., Li20, Li402, Li02) not just reduced formulas.

#### B.4.6 Agent LLM Orchestrator

We implement the agentic discovery policy using DSPy’s ReAct framework *(Khattab et al., 2023; Yao et al., 2022)*. At each oracle query, the orchestrator reasons over the current discovery state and selects actions from a fixed tool set. The agent maintains a composition-indexed buffer of candidate structures that have passed static validity checks; a uniqueness filter is always re-applied to new generations, and structures are cached by hash to avoid duplicate processing.

The LLM has access to (i) a summary of the current buffer, (ii) a bounded evaluation history (most recent 20 oracle queries), and (iii) known stable materials from the phase diagram. Each decision step is limited to 10 ReAct iterations.

##### Available tools.

The orchestrator can invoke the following tools:

- generate_structures: generate candidate structures for specified compositions using the generators defined above (Random or Chemeleon);
- create_structure: directly specify a crystal structure by explicitly defining lattice parameters and atomic positions;
- score_buffer: score buffered structures using a selected scorer (e.g., diversity, MLIP, or LLM-based, as defined above.);
- list_compositions: list compositions in buffer ordered by count or score;
- query_structures: retrieve all, random, top or bottom $k$ scoring structures for a given composition;
- get_buffer_stats: report buffer statistics for situational awareness;
- select_for_evaluation: select a single structure (by specifying a composition and buffer index) for oracle evaluation. This is always the final tool call.

The orchestration system prompt is given in listing 3.

Listing 3: LLM Orchestrator Prompt
⬇
You are an autonomous materials discovery agent.

OBJECTIVE: Find as many NOVEL, UNIQUE, STABLE (or metastable) structures as possible. Use the available tools, then select ONE composition + structure for oracle evaluation.

- Structures with e_above_hull <= stability_tolerance are stable/metastable (SUCCESS!)
- Entries marked [STABLE, NOVEL] in evaluation_history are successful discoveries
- We want to MAXIMIZE the number of novel stable structures found

IMPORTANT: Different structures for the SAME composition can have DIFFERENT stabilities.
- One unstable structure for a composition does NOT mean all structures for that composition are unstable.
- Generators produce many different structures for the same composition

UNIT CELL SIZE MATTERS:
- Compositions are stored by REDUCED formula (e.g., Li20)
- But you can generate different UNIT CELL SIZES: Li20, Li402, Li603, etc.
- These occupy the same position on the phase diagram but are different structures

BUFFER ORGANIZATION:
- Buffer is organized by reduced formula: {composition: [structures]}
- Each structure shows its full formula (unit cell size) and index
- Selection is two-step: pick composition, then pick structure index

WORKFLOW:
1. Decide which composition(s) to explore based on evaluation history
2. Generate or create candidate structures for those compositions
3. Score candidates if needed to prioritize within each composition
4. List compositions and query structures to decide what to evaluate
5. Select ONE composition + structure for oracle evaluation

STRATEGY GUIDANCE:
- If buffer empty/small: generate more structures
- If buffer has candidates: score, query, and select
- Balance exploration (new compositions) vs exploitation (promising ones)

### B.4.7 Filters

All generated structures are passed through inexpensive validity filters prior to oracle evaluation to mirror real discovery pipelines. Specifically, we apply:

- a minimum interatomic distance filter to remove unphysical atomic overlaps. This is set at 0.5 Å between atoms, as per the structural validity evaluation metric used in MatterGen *(Zeni et al., 2025)*.
- a uniqueness filter that filters previously attempted structures (see Section B.3).
- a chemical validity filter based on SMACT constraints. For inter-metallic systems, this is redundant.

These filters are applied uniformly across all policies. We observed no qualitative changes in relative performance when applying these filters to other baselines, or versus not using them, so we do not explicitly compare results with and without the filters.

## Appendix C Further Results

### C.1 Scaling System Complexity

Figure 8 illustrates how the number of valid compositions grows rapidly with system size. For each composition, there exists a vast space of possible crystal structures arising from different lattice symmetries, atomic arrangements, and unit-cell

MADE: Benchmark Environments for Closed-Loop Materials Discovery

sizes. This combinatorial growth in both composition and structure leads to an extremely sparse discovery landscape at larger system sizes, making naive enumeration infeasible and underscoring the need for effective, adaptive search strategies to identify new stable materials.

![img-14.jpeg](img-14.jpeg)
Figure 8. The number of unique compositions that can be explored grows rapidly with system size.

# C.2. Distributional Results and Additional Structural Diversity Metrics

Figure 9 shows distributions of the summary results given in Table 1. The top row gives the discovery metrics while the bottom row gives diversity metrics.

We additionally report a continuous structural discrepancy metric: the average minimum distance (AMD) (Widdowson et al., 2022; Widdowson &amp; Kurlin, 2022). We report the mean pairwise AMD between discovered mSUN structures. Naturally, the random generator produces structures with large discrepancies.

![img-15.jpeg](img-15.jpeg)
Figure 9. Discovery and diversity metric distributions, aggregated over system sizes.

MADE: Benchmark Environments for Closed-Loop Materials Discovery

# C.3. Results Across System Sizes

Figures 10, 11 and 12 show discovery metrics broken down by system size and policy. We report distributions over all metrics per system size in Figures 13, 14 and 15.

![img-16.jpeg](img-16.jpeg)
Figure 10. End-to-end materials discovery performance of different policies on ternary systems. The error bars are standard errors in the mean over 10 systems, each with 5 episodes.

![img-17.jpeg](img-17.jpeg)
Figure 11. End-to-end materials discovery performance of different policies on quaternary systems. The error bars are standard errors in the mean over 10 systems, each with 5 episodes.

MADE: Benchmark Environments for Closed-Loop Materials Discovery

![img-18.jpeg](img-18.jpeg)
Figure 12. End-to-end materials discovery performance of different policies on quinary systems. The error bars are standard errors in the mean over 10 systems, each with 5 episodes.

![img-19.jpeg](img-19.jpeg)
Figure 13. Discovery and diversity metric distributions for experiments on ternary systems.

MADE: Benchmark Environments for Closed-Loop Materials Discovery

![img-20.jpeg](img-20.jpeg)
Figure 14. Discovery and diversity metric distributions for experiments on quaternary systems.

![img-21.jpeg](img-21.jpeg)
Figure 15. Discovery and diversity metric distributions for experiments on quinary systems.