### 4. Feasibility of a Cryptocurrency Market Crash Prediction System

#### 4.1. Model Design and Framework
This phase focuses on designing and testing a system capable of reliably forecasting severe downturns in the cryptocurrency market. We treat crash prediction as a supervised sequence classification task. A Long Short-Term Memory (LSTM) neural network serves as the core architecture, chosen specifically for its capacity to learn and retain long-term temporal dependencies within noisy financial time-series data.

#### 4.2. Data Scope and Target Regimes
The predictive model uses daily data spanning from January 1, 2020, to December 31, 2025. This timeframe is highly relevant as it captures the maturation of the crypto market, including the period surrounding the approval of BTC and ETH ETFs.

Crash regimes—which act as the target variable for the LSTM—are identified using rolling realized volatility thresholds applied to Bitcoin and Ethereum returns. To ensure the system is grounded in actual market stress, the model’s predictive capabilities are specifically evaluated against five distinct, high-impact historical crashes:
1. **March 2020:** The "Black Thursday" pandemic-induced market shock.
2. **May 2021:** The broad downturn triggered by China's mining ban and Tesla's environmental concerns regarding Bitcoin.
3. **May–June 2022:** The systemic fallout from the Terra (LUNA) network collapse.
4. **November 2022:** The liquidity crisis and subsequent bankruptcy of the FTX exchange.
5. **August 2024:** The cross-asset market shock associated with the Yen carry trade unwind.

#### 4.3. Input Features and Integration
To construct a robust predictive model, the LSTM is trained on a multivariate set of rolling financial indicators. This feature set directly integrates quantitative outputs from the earlier stages of the study:
*   **Market Dynamics:** Asset returns and historical realized volatility.
*   **Contagion Metrics:** Time-varying DCC correlations and Diebold-Yilmaz connectedness indices. These inputs provide the model with data on the real-time intensity of spillovers between crypto and traditional equity markets.
*   **External Variables:** Selected macroeconomic indicators and standard technical analysis metrics to account for broader economic conditions and market momentum.

#### 4.4. Training and Performance Evaluation
The model is trained to map these rolling sequential inputs to future market states (crash versus non-crash regimes). Because financial crashes are rare events, the resulting dataset is inherently imbalanced. As a result, relying on standard classification accuracy can be misleading. 

The feasibility and reliability of the prediction system are instead evaluated using metrics designed specifically for imbalanced data. We rely on the Area Under the Receiver Operating Characteristic Curve (AUC-ROC) to measure overall discriminatory power. Additionally, Precision, Recall, and F1-scores will be calculated to assess the model's exactness and completeness. Finally, confusion matrices will be generated to explicitly track the rates of false positives and false negatives, providing a practical assessment of how well the system could function in a live risk-management scenario.
