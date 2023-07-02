# BN Creation Bot

**Bayesian Network** is widely used in risk assessment. 
It is a probabilistic model that describes the probability of an event occurring in a given situation.

With our app, you don't need to know the details of the Bayesian Network to create it.
We instruct *ChatGPT* to guide you through the creation process.

- Check out what is a [Bayesian Network](https://en.wikipedia.org/wiki/Bayesian_network)
- Powered by OpenAI's [ChatGPT](https://www.openai.com/blog/chatgpt/): [GPT Model Used](https://platform.openai.com/docs/models/gpt-3-5)
    - gpt-3.5-turbo
    - gpt-3.5-turbo-0613
- Data Source: [Yahoo Finance](https://finance.yahoo.com/) & [FRED](https://fred.stlouisfed.org/).

## Start
```cmd
cd BN-Creation-Bot
pip install -r requirements.txt
python main.py
```

## How it work
- 

## Limitations
- Currently, only work for market correction

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
│   ├── 2_Visualization.py
├── utils
│   ├── chat_help.py
│   ├── utils.json
│   ├── hist_mc.csv
├── About.py
├── main.py
├── requirements.txt
├── stored_session.py
```