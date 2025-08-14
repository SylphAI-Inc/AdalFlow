# State-of-the-Art AI Agent Benchmark Datasets (2024-2025)

## **SWE Benchmark Datasets**

### **SWE-bench: Core Dataset**

**Generic description**: An evaluation framework consisting of 2,294 software engineering problems drawn from real GitHub issues and corresponding pull requests across 12 popular Python repositories.

SWE-bench evaluates how well language models can solve real-world software engineering tasks. The evaluation process involves:
1. **Input**: Code repository and issue description
2. **Task**: Generate a patch that resolves the described problem
3. **Evaluation**: Test patch against existing test suites and issue requirements
4. **Environment**: Fully containerized evaluation using Docker for reproducible results

### **Latest SWE-bench Variants (2024-2025)**

#### **SWE-bench Verified (August 2024)**
A human-validated subset of 500 high-quality test cases from the original benchmark. Each task has been carefully reviewed and validated by human experts to ensure:
- Clear problem statements without ambiguity
- Correct expected solutions
- Proper test case coverage
- Elimination of data leakage issues

**Evaluation methodology**: Uses the same Docker-based evaluation harness as the original SWE-bench, but with stricter quality controls and human verification of ground truth solutions.

#### **SWE-bench Multimodal (January 2025)**
Extends the original benchmark to include issues with visual elements and UI components. Features:
- Screenshots and visual mockups as part of issue descriptions
- UI/UX related bug reports and feature requests
- Frontend component implementation tasks
- Visual regression testing capabilities

#### **SWE-bench Live (2025)**
A continuously updated benchmark to prevent data contamination. Key features:
- Monthly updates with new GitHub issues
- Automated dataset curation pipeline
- Real-time issue collection from active repositories
- Covers 164 repositories across diverse domains

### **SWE-Lancer: Economic-Driven Benchmark (2024)**
**Generic description**: Real freelance software engineering tasks from Upwork, providing economic context to evaluation.

**Evaluation methodology**:
- Tasks sourced from actual Upwork freelance projects
- Economic value ranges from $50 bug fixes to $32,000 feature implementations
- End-to-end testing using Playwright-powered browser automation
- Includes both technical implementation and project management assessment
- Success measured by client acceptance criteria rather than just unit tests

### **Expansion Opportunities for AdalFlow**

Could expand to take not only issues but also pull requests labeled as features for testing CLI's capability of implementing actual features and not just issues. We could start with AdalFlow and see if it can implement core features given design documents in Notion. Open source libraries like PyTorch already provide specific documentation and detail for each feature.

For these repositories, we can create a similar dataset with the base commit before the feature was merged, provide the documentation and details, and evaluate on the tests part of the feature PR.


## **MLE Benchmark (Kaggle Competitions)**

### **MLE-bench: OpenAI's Machine Learning Engineering Benchmark (2024)**

**Generic description**: A benchmark for measuring how well AI agents perform at machine learning engineering, constructed using a collection of 75 ML engineering competitions sourced from Kaggle.

**Evaluation methodology**:
1. **Environment Setup**: Each competition runs in an isolated Docker container with standardized dependencies
2. **Data Access**: Agents receive training data and competition description, must produce final submission
3. **Evaluation Pipeline**: 
   - Agents have access to local evaluation tools for validation
   - Final submissions scored against held-out test sets
   - Performance measured using Kaggle's medal system (Bronze/Silver/Gold)
4. **Resource Management**: Computational limits and timeouts enforced per competition
5. **Multi-domain Testing**: Covers NLP, computer vision, signal processing, and tabular data tasks

**Key Infrastructure**:
- Preparation scripts split publicly available training sets into new train/test divisions
- Each competition includes problem description, dataset, local evaluation tools, and automated grading
- Standardized agent interfaces across all 75 competitions
- Reproducible evaluation with consistent environment specifications

### **Latest Kaggle AI Agent Competitions (2024-2025)**

#### **ARC Prize 2024 & 2025**
**Generic description**: Evaluation of AI systems on novel reasoning tasks requiring efficient learning of new skills.

**Evaluation methodology**:
- Tests on abstract reasoning corpus requiring pattern recognition and rule inference
- Computational budget constraints ($50 for 120 evaluation tasks)
- Measures both performance accuracy and efficiency per task
- Human baseline comparison for intelligence measurement

#### **Kaggle Game Arena (2024)**
**Generic description**: Platform for head-to-head AI model competition in strategic games.

**Evaluation framework**:
- Real-time strategic game competitions (Go, poker, video games)
- ELO rating system for skill measurement
- Head-to-head matchmaking and tournament structures
- Dynamic leaderboards with continuous evaluation

### **Expansion Opportunities for AdalFlow**

Pull Kaggle competitions and split public dataset into training, validation and testing dataset and evaluate model submission performance as medals. From the evaluation, a base Docker environment is created for each agent that can be used for each of the competitions (common dependencies). Similarly, can integrate cloud compute or provided Docker environments to execute code and submit against model competitions.

The agent should be able to compute using a docker container on the cloud or locally. The cloud would be preferable for parallelizing training and testin.

## **Paper Implementation Datasets**

### **Paper2Code: Automating Code Generation from Scientific Papers**

**Link**: https://github.com/going-doer/Paper2Code

**Generic description**: A benchmark that evaluates agents' ability to implement papers, which can be an important aspect of AI development.

#### **Paper2CodeBench (2024)**
**Dataset Construction Methodology**:
1. **Paper Collection**: Uses OpenReview API to gather accepted papers from ICLR, ICML, and NeurIPS 2024
2. **Filtering Criteria**: 
   - Code availability requirement
   - Token limit <70,000 to ensure LLM processing feasibility
   - Quality screening using GPT-4o model-based evaluation
3. **Final Selection**: Top 30 papers from each venue (90 total papers)

**Evaluation Framework**:
- **Input**: Research paper PDF and any supplementary materials
- **Task**: Generate complete, functional code repository implementing the paper's methodology
- **Assessment**: Multi-stage evaluation including:
  - Code functionality and correctness
  - Implementation completeness
  - Adherence to paper specifications
  - Reproducibility of reported results

#### **PaperBench Code-Dev**
**Generic description**: 20 papers from ICML 2024 with human-annotated evaluation rubrics.

**Evaluation methodology**:
- Paper-specific rubrics created by human experts
- LLM-based evaluation using structured assessment criteria
- Focus on implementation correctness rather than just code generation
- Rubrics cover algorithm accuracy, code quality, and experimental reproducibility

### **MLR-Bench: Machine Learning Research Benchmark (2024)**

**Generic description**: A comprehensive benchmark for evaluating AI agents on open-ended machine learning research tasks.

**Evaluation Infrastructure**:
1. **MLR-Judge**: Automated evaluation framework using LLM-based reviewers with structured rubrics
2. **MLR-Agent**: Modular agent scaffold providing standardized interface for task completion
3. **Multi-stage Assessment**: Research proposal, methodology, implementation, and result analysis

**Task Methodology**:
- **Task Categories**: 9 core ML topics covering diverse research areas
- **Open-ended Format**: Tasks require creative problem-solving rather than just implementation
- **Research Pipeline**: Full research cycle from problem formulation to experimental validation
- **Peer Review Simulation**: LLM-based reviewers assess work quality using academic standards

### **Expansion Opportunities for AdalFlow**

**(Implementing papers)**

**Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning**

They evaluated agents' ability to implement papers which can be an important aspect.

We construct a new benchmark dataset, which we refer to as Paper2CodeBench. Specifically, we collect the accepted papers from recent machine learning venues (such as ICLR, ICML, and NeurIPS 2024) with the OpenReview API, and filter them based on the availability of code with its total number of tokens less than 70,000, to ensure the full repository remains within reasonable processing limits of modern LLMs for generation and evaluation. Also, to maintain the benchmark quality, we perform model-based evaluation with GPT-4o on all the collected repositories and select the top 30 from each venue, resulting in a total of 90 papers.

In addition to Paper2CodeBench, we also use the recently released PaperBench Code-Dev, which consists of 20 papers from ICML 2024 with paper-specific rubrics annotated by humans. In particular, those rubrics are used to judge the correct implementation based on LLM-based evaluation.

## **Additional State-of-the-Art Benchmarks (2024-2025)**

### **Agent-Specific Benchmarks**

#### **AgentBoard (NeurIPS 2024)**
**Generic description**: Analytical evaluation board for multi-turn LLM agents focusing on complex interaction scenarios.

**Evaluation methodology**:
- Multi-turn conversation evaluation with state persistence
- Partially-observable environment maintenance across interactions
- Long-horizon task completion assessment
- Comprehensive evaluation across diverse agent capabilities

#### **WebAgent for Real-World Web Tasks (ICLR 2024)**
**Generic description**: LLM-driven agent evaluation on real website interaction tasks.

**Evaluation framework**:
- Real website interaction using browser automation
- Natural language instruction parsing and execution
- Task decomposition into canonical sub-instructions
- HTML document processing and summarization capabilities
- End-to-end task completion measurement

#### **MetaGPT Multi-Agent Framework (ICLR 2024)**
**Generic description**: Meta-programming framework for LLM-based multi-agent collaborations.

**Evaluation approach**:
- Multi-agent coordination and communication assessment
- Workflow efficiency measurement in collaborative tasks
- Human workflow integration and optimization
- Complex project completion evaluation

## **How AdalFlow Datasets Could Be Constructed**

Based on the analysis of state-of-the-art benchmarks, AdalFlow could construct datasets in several key areas:

### **1. SWE-bench Style Implementation**
**Evaluation methodology**:
- Use AdalFlow's own GitHub issues and feature requests as source material
- Create base commits before features were merged to establish clean starting points
- Provide design documents from Notion as contextual requirements
- Evaluate against existing test suites and integration tests
- Include Docker containerization for reproducible evaluation environments
- Test both bug fixes and feature implementations

### **2. MLE-bench Integration**
**Evaluation framework**:
- Curate machine learning competitions specifically relevant to AdalFlow's optimization capabilities
- Create standardized Docker environments for reproducible evaluation across different tasks
- Focus evaluation on prompt optimization, few-shot learning, and auto-differentiation tasks
- Implement cloud compute integration for scalable evaluation infrastructure
- Develop metrics for measuring optimization efficiency alongside task performance

### **3. Paper Implementation Focus**
**Evaluation approach**:
- Target papers related to auto-optimization, prompt engineering, and LLM workflow optimization
- Use AdalFlow's research on "Auto-Differentiating Any LLM Workflow" as baseline implementation
- Create specialized rubrics for evaluating optimization algorithm implementations
- Focus on recent LLM workflow optimization papers from top-tier venues
- Implement automated testing for optimization convergence and performance improvement
