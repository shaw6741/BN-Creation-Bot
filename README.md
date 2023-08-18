# BN Creation Bot
**Bayesian Network** is a powerful tool for risk assessment and probabilistic modeling. It allows you to understand the probability of events occurring in various situations.

The WebApp simplifies the process of creating a Bayesian Network. You don't need to know the technical details and coding for Bayesian Network. The intuitive interface powered by <u>ChatGPT</u> & <u>Streamlit</u> will guide you through the entire creation process.

- Check out what is a [Bayesian Network](https://en.wikipedia.org/wiki/Bayesian_network)
- [GPT Model Used](https://platform.openai.com/docs/models/gpt-3-5): gpt-3.5-turbo & gpt-3.5-turbo-0613
- Data Source: [Yahoo Finance](https://finance.yahoo.com/) & [FRED](https://fred.stlouisfed.org/).

## Get Start
```
cd BN-Creation-Bot
pip install -r requirements.txt
python main.py
```

## How it work
- **Prompt Engineering**: ensure that ChatGPT controls the overall process, gathers necessary information, and provides formatted output for easy extraction and code integration.
- **Communication**: Users interact with ChatGPT to provide inputs such as triggers, controls, risk events, mitigators, consequences, and edges. Conversational interaction allows users to seek explanations and refine the network.
- **Information Analysis**: ChatGPT analyzes and summarizes the provided information, extracting relevant dependencies and relationships between variables in a formatted manner. This enables further processing and network creation.
- **Network Creation**: Once the necessary info has been acquired and analyzed, the app generates a Bayesian Network using the formatted answer from ChatGPT.
- **Visualization & Exploration**: offers various visualizations to provide insights into the network and probabilistic relationships between variables.


## Limitations
- Currently, only work for market correction/financial data/discrete data

## File Structure
```bash
├── baynet
│   ├── bayesian_model.py
├── engine
│   ├── Engine.py
│   ├── conversation.json
│   ├── node_dic.json
├── pages
│   ├── 1_Chat.py
│   ├── 2_ProbChat.py
│   ├── 3_Visualization.py
├── utils
│   ├── chat_help.py
│   ├── cond_prob_chat.py
│   ├── prior_prob_chat.py
│   ├── utils.py
│   ├── visual_help.py
│   ├── hist_mc.csv
├── About.py
├── main.py
├── requirements.txt
```