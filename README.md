# Disclaimer
I am a high school student, and definitely not a financial advisor. Feel free to use this project but I cannot guarantee any financial success. This was mostly just a fun excuse to try Proximal Policy Optimization (PPO) and dive deeper into reinforcement learning.


# Performance
I did not perform very rigorous tests, but by observing the graphs of total worth over time we can see a significant improvement from the untrained model, which suggests that the PPO algorithm is effective.  
  
Before training:  
![Figure_2](https://github.com/user-attachments/assets/6bf6d4e7-5207-46c9-90a6-075bb5b02ddc)

After training:  
![Figure_3(PostTrain)](https://github.com/user-attachments/assets/5e39e09c-886f-461c-9acb-77df8ab705b1)  

# Architecture & Design
As mentioned previously, the model uses PPO with reinforcement learning. The agent itself is comprised of an action agent that decides which stock to interact with (or to stop trading for the day) and a seperate quantity agent that controls the amount of stock to buy or sell.  
  
More on PPO can be found here: https://arxiv.org/abs/1707.06347  

Note that tuning the hyperparameters may offer differing experience based on individual hardware requirements. If performance is the goal, tuning the hyperparameters or modifying the architecture may prove more beneficial depending on your hardware.

# Usage
Firstly, in order to get the data, get a Tiingo API token and copy it into the "\[YOUR_API_TOKEN\]" part of the API request (In the API.py file). Change the stocks under the if \_\_name\_\_ == "\_\_main\_\_" to anything you want the model to be able to trade, then run the API file to get the data (It should be saved to stock_data.json).  

After the data is acquired, change the stocks in the main file under the parameters to match those in the data, and run the main file after making appropriate modifications (uncomment testing lines, etc.).


