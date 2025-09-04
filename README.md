# MMA: Cross-Domain Knowledge Integration via Mixture of Multi-Domain Agents
Mixture of Multi-domain Agents (MMA) dynamically combines the outputs of general-purpose and domain-specific models to enhance their performance on complex, domain‑specific tasks. MMA formulates the integration process as a search problem, using Monte Carlo Tree Search (MCTS) to find the path that optimally harmonizes the respective strengths of different models in generalization and domain-specific knowledge. 

# Environment
```
conda create -n mma python=3.10
conda activate mma
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

# Quick start
1.Download data to directory.

`mkdir ./data/JEJS`

Download the `3-7.json` file from [LawBench](https://github.com/open-compass/LawBench) to `JEJS`.

2.Build your FewShot prompt.

In the `prompts/JEJS directory`, complete all json files with examples.

Or you can skip adding examples and proceed directly to the next step.

3.Run.

`mkdir ./output/JEJS`

`bash scripts/run_jejs_generator.sh`
