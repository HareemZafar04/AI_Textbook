---
sidebar_label: Finance Applications
---

# Finance Applications of AI

Artificial Intelligence has transformed the financial sector by enabling more accurate risk assessment, fraud detection, algorithmic trading, and personalized financial services. This section explores the various applications of AI in finance and their impact on the industry.

## Overview of AI in Finance

AI in finance encompasses a wide range of applications including algorithmic trading, credit scoring, fraud detection, robo-advisory, customer service automation, and risk management. The technology leverages vast amounts of financial data to make faster, more accurate decisions and provide personalized services.

## Algorithmic Trading

AI algorithms can analyze market data and execute trades at high speeds with minimal human intervention.

### Market Prediction with Technical Indicators
```python
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import talib

class AlgorithmicTradingAI:
    def __init__(self, symbol="AAPL", period="2y"):
        self.symbol = symbol
        self.period = period
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        
    def fetch_data(self):
        """Fetch stock data from Yahoo Finance"""
        stock = yf.Ticker(self.symbol)
        data = stock.history(period=self.period)
        return data
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators as features"""
        # Moving averages
        df['SMA_10'] = talib.SMA(df['Close'], timeperiod=10)
        df['SMA_30'] = talib.SMA(df['Close'], timeperiod=30)
        df['EMA_10'] = talib.EMA(df['Close'], timeperiod=10)
        
        # Relative Strength Index (RSI)
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        
        # Bollinger Bands
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'])
        
        # MACD
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'])
        
        # Price rate of change
        df['ROC'] = talib.ROC(df['Close'], timeperiod=10)
        
        # Commodity Channel Index
        df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        # Calculate daily returns
        df['daily_return'] = df['Close'].pct_change()
        
        # Create target variable (1 for price increase, 0 for decrease)
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for machine learning"""
        # Select features for model
        feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_10', 'SMA_30', 'EMA_10', 'RSI', 'ROC',
            'MACD', 'MACD_signal', 'CCI'
        ]
        
        # Remove rows with NaN values
        df = df.dropna()
        
        X = df[feature_cols].values
        y = df['target'].values
        
        return X, y, df
    
    def train(self):
        """Train the algorithmic trading model"""
        # Fetch and preprocess data
        data = self.fetch_data()
        data = self.calculate_technical_indicators(data)
        X, y, self.processed_data = self.prepare_features(data)
        
        # Split data (last 20% for testing)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained successfully!")
        print(f"Prediction accuracy: {accuracy:.4f}")
        
        return accuracy
    
    def predict_next_move(self):
        """Predict if the stock will go up or down tomorrow"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Get the latest data row
        latest_data = self.processed_data.iloc[-1:]
        
        # Extract the same features used in training
        feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_10', 'SMA_30', 'EMA_10', 'RSI', 'ROC',
            'MACD', 'MACD_signal', 'CCI'
        ]
        
        X = latest_data[feature_cols].values
        
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]
        
        direction = "UP" if prediction == 1 else "DOWN"
        confidence = max(probability)
        
        return {
            'prediction': direction,
            'confidence': confidence,
            'probabilities': {
                'down': probability[0],
                'up': probability[1]
            }
        }

# Example usage (uncomment when needed)
# trader = AlgorithmicTradingAI(symbol="AAPL")
# accuracy = trader.train()
# prediction = trader.predict_next_move()
# print(f"Next day prediction: {prediction['prediction']} with {prediction['confidence']:.2%} confidence")
```

### Portfolio Optimization
```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf

class PortfolioOptimizer:
    def __init__(self, symbols, period="1y"):
        self.symbols = symbols
        self.period = period
        self.data = None
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        
    def fetch_data(self):
        """Fetch historical data for all symbols"""
        data = yf.download(self.symbols, period=self.period)['Adj Close']
        self.data = data
        self.returns = data.pct_change().dropna()
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        
    def portfolio_performance(self, weights):
        """Calculate portfolio return and volatility"""
        returns = np.sum(self.mean_returns * weights) * 252  # Annualized
        volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        return returns, volatility
    
    def negative_sharpe_ratio(self, weights):
        """Calculate negative Sharpe ratio (to minimize)"""
        p_returns, p_volatility = self.portfolio_performance(weights)
        return -(p_returns - 0.02) / p_volatility  # Assuming 2% risk-free rate
    
    def optimize_portfolio(self):
        """Find optimal portfolio weights"""
        if self.data is None:
            self.fetch_data()
        
        num_assets = len(self.symbols)
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: weights between 0 and 1 (no shorting)
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Initial guess: equal weights
        initial_weights = num_assets * [1. / num_assets]
        
        # Optimize
        result = minimize(
            self.negative_sharpe_ratio,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
    
    def get_portfolio_metrics(self, weights):
        """Calculate portfolio metrics for given weights"""
        returns, volatility = self.portfolio_performance(weights)
        sharpe_ratio = (returns - 0.02) / volatility  # Assuming 2% risk-free rate
        
        return {
            'expected_annual_return': returns,
            'annual_volatility': volatility,
            'sharpe_ratio': sharpe_ratio
        }

# Example usage (uncomment when needed)
# symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
# optimizer = PortfolioOptimizer(symbols)
# optimal_weights = optimizer.optimize_portfolio()
# metrics = optimizer.get_portfolio_metrics(optimal_weights)
# 
# print("Optimal Portfolio Weights:")
# for i, symbol in enumerate(symbols):
#     print(f"  {symbol}: {optimal_weights[i]:.4f}")
# 
# print(f"\nPortfolio Metrics:")
# print(f"  Expected Annual Return: {metrics['expected_annual_return']:.4f}")
# print(f"  Annual Volatility: {metrics['annual_volatility']:.4f}")
# print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
```

## Credit Scoring

AI models can assess creditworthiness more accurately by analyzing multiple data points.

### Credit Risk Assessment Model
```python
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

class CreditRiskModel:
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = [
            'age', 'income', 'employment_length', 'debt_to_income',
            'credit_utilization', 'num_credit_accounts', 
            'num_delinquencies', 'num_defaults', 
            'loan_amount', 'loan_purpose_encoded', 
            'home_ownership_encoded', 'education_encoded'
        ]
        self.is_trained = False
    
    def generate_synthetic_data(self, n_samples=5000):
        """Generate synthetic credit data for demonstration"""
        np.random.seed(42)
        
        data = {
            'age': np.random.normal(40, 12, n_samples).clip(18, 80),
            'income': np.random.lognormal(np.log(50000), 0.7, n_samples),
            'employment_length': np.random.exponential(5, n_samples).clip(0, 30),
            'debt_to_income': np.random.beta(2, 5, n_samples).clip(0, 1),
            'credit_utilization': np.random.beta(2, 3, n_samples).clip(0, 1),
            'num_credit_accounts': np.random.poisson(4, n_samples).clip(1, 15),
            'num_delinquencies': np.random.poisson(0.2, n_samples).clip(0, 5),
            'num_defaults': np.random.binomial(1, 0.05, n_samples),
            'loan_amount': np.random.lognormal(np.log(15000), 0.8, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Encode categorical variables
        loan_purposes = ['debt_consolidation', 'home_improvement', 'major_purchase', 'other']
        home_ownership = ['rent', 'own', 'mortgage', 'other']
        education_levels = ['high_school', 'college', 'grad_school', 'phd']
        
        df['loan_purpose'] = np.random.choice(loan_purposes, n_samples)
        df['home_ownership'] = np.random.choice(home_ownership, n_samples)
        df['education'] = np.random.choice(education_levels, n_samples)
        
        # Encode categorical variables
        df['loan_purpose_encoded'] = pd.Categorical(df['loan_purpose']).codes
        df['home_ownership_encoded'] = pd.Categorical(df['home_ownership']).codes
        df['education_encoded'] = pd.Categorical(df['education']).codes
        
        # Simulate credit risk based on features
        risk_score = (
            -0.1 * (df['age'] - 40) / 10 +
            -0.2 * (df['income'] / 10000 - 5) +
            -0.15 * df['employment_length'] / 10 +
            0.4 * df['debt_to_income'] +
            0.3 * df['credit_utilization'] +
            0.05 * df['num_credit_accounts'] +
            0.3 * df['num_delinquencies'] +
            0.4 * df['num_defaults'] +
            0.1 * (df['loan_amount'] / 10000 - 1.5) +
            0.05 * df['loan_purpose_encoded'] +
            0.05 * df['home_ownership_encoded'] +
            0.02 * df['education_encoded'] +
            np.random.normal(0, 0.2, n_samples)
        )
        
        # Convert to binary default classification
        df['default'] = (risk_score > np.percentile(risk_score, 70)).astype(int)
        
        return df
    
    def train(self):
        """Train the credit risk model"""
        df = self.generate_synthetic_data()
        
        # Prepare features and target
        X = df[self.feature_names]
        y = df['default']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = self.model.score(X_test_scaled, y_test)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Credit Risk Model trained successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC Score: {auc_score:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return self.model
    
    def assess_credit_risk(self, applicant_data):
        """Assess credit risk for a new applicant"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Prepare applicant data
        applicant_df = pd.DataFrame([applicant_data])
        X = self.scaler.transform(applicant_df[self.feature_names])
        
        # Get prediction and probability
        default_probability = self.model.predict_proba(X)[0][1]
        risk_decision = self.model.predict(X)[0]
        
        # Risk categories
        if default_probability < 0.2:
            risk_level = "Low"
        elif default_probability < 0.5:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return {
            'risk_level': risk_level,
            'default_probability': default_probability,
            'credit_approved': risk_decision == 0,  # 0 means no default, so approve
            'credit_score': int((1 - default_probability) * 850),  # Convert to credit score range
            'recommendation': f"Applicant is {risk_level} risk with {default_probability:.2%} default probability"
        }

# Example usage
credit_model = CreditRiskModel()
model = credit_model.train()

# Example applicant data
applicant = {
    'age': 35,
    'income': 75000,
    'employment_length': 8,
    'debt_to_income': 0.25,
    'credit_utilization': 0.35,
    'num_credit_accounts': 5,
    'num_delinquencies': 0,
    'num_defaults': 0,
    'loan_amount': 25000,
    'loan_purpose_encoded': 0,  # debt consolidation
    'home_ownership_encoded': 2,  # mortgage
    'education_encoded': 1  # college
}

risk_assessment = credit_model.assess_credit_risk(applicant)
print(f"\nCredit Risk Assessment:")
print(f"  Risk Level: {risk_assessment['risk_level']}")
print(f"  Default Probability: {risk_assessment['default_probability']:.2%}")
print(f"  Credit Score: {risk_assessment['credit_score']}")
print(f"  Approved: {risk_assessment['credit_approved']}")
print(f"  Recommendation: {risk_assessment['recommendation']}")
```

## Fraud Detection

AI systems can identify potentially fraudulent transactions in real-time.

### Anomaly Detection for Fraud
```python
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

class FraudDetectionModel:
    def __init__(self):
        self.model = IsolationForest(
            contamination=0.1,  # Expected fraud rate
            random_state=42,
            n_estimators=100
        )
        self.feature_names = [
            'transaction_amount', 'account_balance', 'time_since_last_transaction',
            'day_of_week', 'hour_of_day', 'merchant_category_encoded',
            'location_risk_score', 'card_age_days', 'daily_transaction_count',
            'weekly_transaction_count', 'monthly_transaction_count'
        ]
        self.is_trained = False
    
    def generate_synthetic_transaction_data(self, n_samples=10000):
        """Generate synthetic transaction data with fraud examples"""
        np.random.seed(42)
        
        # Normal transactions
        n_normal = int(n_samples * 0.95)
        n_fraud = n_samples - n_normal
        
        data = {
            'transaction_amount': np.concatenate([
                np.random.lognormal(np.log(50), 0.8, n_normal),  # Normal transactions
                np.random.lognormal(np.log(200), 1.2, n_fraud)    # Fraudulent transactions (higher amounts)
            ]),
            
            'account_balance': np.concatenate([
                np.random.lognormal(np.log(5000), 0.5, n_normal),  # Normal balances
                np.random.lognormal(np.log(500), 0.8, n_fraud)    # Lower balances for fraud
            ]),
            
            'time_since_last_transaction': np.concatenate([
                np.random.exponential(2, n_normal),  # Normal: more frequent transactions
                np.random.exponential(10, n_fraud)   # Fraud: longer gaps
            ]).clip(0, 100),
            
            'day_of_week': np.concatenate([
                np.random.randint(0, 7, n_normal),
                np.random.randint(0, 7, n_fraud)
            ]),
            
            'hour_of_day': np.concatenate([
                np.random.randint(0, 24, n_normal),
                np.random.randint(0, 24, n_fraud)
            ]),
            
            'merchant_category_encoded': np.concatenate([
                np.random.randint(0, 10, n_normal),
                np.random.randint(0, 10, n_fraud)
            ]),
            
            'location_risk_score': np.concatenate([
                np.random.beta(2, 5, n_normal),  # Lower risk for normal
                np.random.beta(5, 2, n_fraud)    # Higher risk for fraud
            ]),
            
            'card_age_days': np.concatenate([
                np.random.exponential(365, n_normal),  # Older cards for normal
                np.random.exponential(30, n_fraud)     # Newer cards for fraud
            ]).clip(0, 2000),
            
            'daily_transaction_count': np.concatenate([
                np.random.poisson(2, n_normal),  # Normal: 2 per day on average
                np.random.poisson(5, n_fraud)    # Fraud: 5 per day on average
            ]).clip(0, 20),
            
            'weekly_transaction_count': np.concatenate([
                np.random.poisson(10, n_normal),
                np.random.poisson(25, n_fraud)
            ]).clip(0, 100),
            
            'monthly_transaction_count': np.concatenate([
                np.random.poisson(30, n_normal),
                np.random.poisson(75, n_fraud)
            ]).clip(0, 300)
        }
        
        df = pd.DataFrame(data)
        
        # Create fraud labels (1 for fraud, 0 for normal)
        labels = [0] * n_normal + [1] * n_fraud
        df['is_fraud'] = labels
        
        return df
    
    def train(self):
        """Train the fraud detection model"""
        df = self.generate_synthetic_transaction_data()
        
        # Prepare features
        X = df[self.feature_names]
        y = df['is_fraud']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(X_train)
        self.is_trained = True
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        # Convert IsolationForest results: -1 = anomaly (fraud), 1 = normal
        # Convert to: 1 = fraud, 0 = normal
        y_pred = (y_pred == -1).astype(int)
        
        print(f"Fraud Detection Model trained successfully!")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return self.model
    
    def detect_fraud(self, transaction_data):
        """Detect if a transaction is fraudulent"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Prepare transaction data
        transaction_df = pd.DataFrame([transaction_data])
        X = transaction_df[self.feature_names]
        
        # Get anomaly score (negative scores indicate anomalies/fraud)
        anomaly_score = self.model.decision_function(X)[0]
        fraud_probability = self.model.score_samples(X)[0]
        
        # Predict (1 for fraud, 0 for normal)
        is_fraud = self.model.predict(X)[0] == -1
        
        return {
            'is_fraud': bool(is_fraud),
            'anomaly_score': anomaly_score,
            'fraud_probability': fraud_probability,
            'action': "FLAG" if is_fraud else "APPROVE"
        }

# Example usage
fraud_detector = FraudDetectionModel()
model = fraud_detector.train()

# Example transaction data
transaction = {
    'transaction_amount': 2500,  # Unusually high amount
    'account_balance': 100,      # Low balance relative to transaction
    'time_since_last_transaction': 0.1,  # Very recent last transaction
    'day_of_week': 3,  # Thursday
    'hour_of_day': 2,  # 2 AM
    'merchant_category_encoded': 7,  # Atypical merchant
    'location_risk_score': 0.9,  # High risk location
    'card_age_days': 5,  # Very new card
    'daily_transaction_count': 10,  # High daily activity
    'weekly_transaction_count': 35,  # High weekly activity
    'monthly_transaction_count': 120  # High monthly activity
}

fraud_result = fraud_detector.detect_fraud(transaction)
print(f"\nFraud Detection Result:")
print(f"  Is Fraud: {fraud_result['is_fraud']}")
print(f"  Anomaly Score: {fraud_result['anomaly_score']:.4f}")
print(f"  Action: {fraud_result['action']}")
```

## Robo-Advisory

AI-powered investment platforms that provide automated portfolio management.

### Robo-Advisor System
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class RoboAdvisor:
    def __init__(self):
        self.risk_models = {}
        self.portfolio_recommender = LinearRegression()
        self.is_trained = False
        
    def generate_customer_data(self, n_customers=1000):
        """Generate synthetic customer data for robo-advisor"""
        np.random.seed(42)
        
        data = {
            'customer_id': range(1, n_customers + 1),
            'age': np.random.normal(45, 15, n_customers).clip(18, 80),
            'income': np.random.lognormal(np.log(60000), 0.8, n_customers),
            'net_worth': np.random.lognormal(np.log(200000), 1.0, n_customers),
            'investment_experience': np.random.choice([1, 2, 3, 4, 5], n_customers, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
            'risk_tolerance': np.random.choice([1, 2, 3, 4, 5], n_customers, p=[0.1, 0.2, 0.4, 0.2, 0.1]),  # 1-5 scale
            'time_horizon_years': np.random.gamma(5, 5, n_customers).clip(1, 40),
            'financial_goals': np.random.choice(['retirement', 'wealth_growth', 'income', 'preservation'], n_customers),
            'current_investment_allocation': np.random.dirichlet([1, 1, 1, 1], n_customers)  # [stocks, bonds, real_estate, cash]
        }
        
        df = pd.DataFrame(data)
        
        # Calculate risk capacity (how much risk one can afford to take)
        df['risk_capacity'] = (
            0.5 * (df['time_horizon_years'] / 20) +
            0.3 * (np.log(df['net_worth']) / 12) +  # Higher net worth, higher capacity
            0.2 * (df['investment_experience'] / 5)
        ).clip(0, 1)
        
        # Calculate risk profile combining tolerance and capacity
        df['risk_score'] = (
            0.6 * df['risk_tolerance'] / 5 +  # Risk tolerance (60% weight)
            0.4 * df['risk_capacity']          # Risk capacity (40% weight)
        )
        
        return df
    
    def generate_asset_returns(self, n_assets=10, n_periods=120):
        """Generate synthetic asset returns"""
        np.random.seed(42)
        
        # Define asset characteristics (expected return, volatility)
        assets = {
            'Large Cap Stocks': {'expected_return': 0.08, 'volatility': 0.15},
            'Small Cap Stocks': {'expected_return': 0.10, 'volatility': 0.20},
            'International Stocks': {'expected_return': 0.07, 'volatility': 0.18},
            'Bonds': {'expected_return': 0.03, 'volatility': 0.05},
            'REITs': {'expected_return': 0.06, 'volatility': 0.12},
            'Commodities': {'expected_return': 0.05, 'volatility': 0.25},
            'Cash': {'expected_return': 0.01, 'volatility': 0.01},
            'Emerging Markets': {'expected_return': 0.09, 'volatility': 0.22},
            'Bonds (Intl)': {'expected_return': 0.04, 'volatility': 0.07},
            'TIPS': {'expected_return': 0.02, 'volatility': 0.04}
        }
        
        asset_names = list(assets.keys())
        returns = pd.DataFrame(index=range(n_periods), columns=asset_names)
        
        for asset in asset_names:
            mean_return = assets[asset]['expected_return'] / 12  # Monthly
            volatility = assets[asset]['volatility'] / np.sqrt(12)  # Monthly
            
            # Generate returns with some correlation
            base_return = np.random.normal(mean_return, volatility, n_periods)
            returns[asset] = base_return
        
        return returns, assets
    
    def train(self):
        """Train the robo-advisor system"""
        customer_data = self.generate_customer_data()
        asset_returns, self.asset_info = self.generate_asset_returns()
        
        # For each customer, determine optimal allocation based on risk score
        allocations = []
        
        for idx, customer in customer_data.iterrows():
            risk_score = customer['risk_score']
            
            # Define allocation strategy based on risk score
            if risk_score <= 0.3:  # Conservative
                allocation = [0.2, 0.1, 0.1, 0.4, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0]  # More bonds/cash
            elif risk_score <= 0.6:  # Moderate
                allocation = [0.3, 0.15, 0.1, 0.25, 0.1, 0.05, 0.05, 0.0, 0.0, 0.0]
            else:  # Aggressive
                allocation = [0.4, 0.2, 0.15, 0.1, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0]  # More stocks
            
            allocations.append(allocation)
        
        customer_data['optimal_allocation'] = allocations
        self.customer_data = customer_data
        
        # Train model to predict allocations based on customer features
        feature_cols = ['age', 'income', 'net_worth', 'risk_tolerance', 'time_horizon_years']
        X = customer_data[feature_cols]
        y = customer_data['risk_score']
        
        self.portfolio_recommender.fit(X, y)
        self.is_trained = True
        
        print("Robo-Advisor trained successfully!")
        print(f"Learned from {len(customer_data)} customer profiles")
        
        return self.portfolio_recommender
    
    def recommend_portfolio(self, customer_profile):
        """Recommend portfolio allocation for a customer"""
        if not self.is_trained:
            raise ValueError("Robo-advisor must be trained first")
        
        # Predict risk score
        features = np.array([[
            customer_profile['age'],
            customer_profile['income'],
            customer_profile['net_worth'],
            customer_profile['risk_tolerance'],
            customer_profile['time_horizon_years']
        ]])
        
        predicted_risk_score = self.portfolio_recommender.predict(features)[0]
        
        # Determine allocation based on risk score
        if predicted_risk_score <= 0.3:  # Conservative
            allocation = [0.2, 0.1, 0.1, 0.4, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0]
        elif predicted_risk_score <= 0.6:  # Moderate
            allocation = [0.3, 0.15, 0.1, 0.25, 0.1, 0.05, 0.05, 0.0, 0.0, 0.0]
        else:  # Aggressive
            allocation = [0.4, 0.2, 0.15, 0.1, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0]
        
        # Get asset names
        asset_names = list(self.asset_info.keys())
        
        # Create portfolio recommendation
        portfolio = []
        for i, asset in enumerate(asset_names):
            portfolio.append({
                'asset': asset,
                'allocation': allocation[i],
                'expected_return': self.asset_info[asset]['expected_return'],
                'volatility': self.asset_info[asset]['volatility']
            })
        
        # Calculate expected portfolio return and volatility
        expected_return = sum(p['allocation'] * p['expected_return'] for p in portfolio)
        volatility = np.sqrt(sum((p['allocation'] * p['volatility'])**2 for p in portfolio))  # Simplified
        
        return {
            'predicted_risk_score': predicted_risk_score,
            'risk_level': 'Conservative' if predicted_risk_score <= 0.3 else 
                         'Moderate' if predicted_risk_score <= 0.6 else 'Aggressive',
            'portfolio': portfolio,
            'expected_annual_return': expected_return,
            'expected_annual_volatility': volatility
        }

# Example usage
robo_advisor = RoboAdvisor()
model = robo_advisor.train()

# Example customer profile
customer = {
    'age': 35,
    'income': 80000,
    'net_worth': 150000,
    'risk_tolerance': 4,  # On a scale of 1-5
    'time_horizon_years': 25,
    'investment_experience': 3  # On a scale of 1-5
}

portfolio_recommendation = robo_advisor.recommend_portfolio(customer)
print(f"\nPortfolio Recommendation for Customer:")
print(f"Risk Level: {portfolio_recommendation['risk_level']}")
print(f"Expected Annual Return: {portfolio_recommendation['expected_annual_return']:.2%}")
print(f"Expected Annual Volatility: {portfolio_recommendation['expected_annual_volatility']:.2%}")

print("\nRecommended Allocation:")
for asset in portfolio_recommendation['portfolio']:
    if asset['allocation'] > 0:
        print(f"  {asset['asset']}: {asset['allocation']:.1%} "
              f"(Exp. return: {asset['expected_return']:.1%})")
```

## Regulatory Compliance

AI can help financial institutions ensure compliance with regulations.

### Compliance Monitoring System
```python
import datetime
import re

class ComplianceMonitor:
    def __init__(self):
        self.regulations = {
            'KYC': {
                'requirements': ['customer_id_verification', 'address_verification', 'source_of_wealth'],
                'required_frequency': {'customer_id_verification': 365, 'address_verification': 180}
            },
            'AML': {
                'requirements': ['transaction_monitoring', 'suspicious_activity_reporting', 'customer_due_diligence'],
                'thresholds': {'suspicious_transaction': 10000}
            },
            'GDPR': {
                'requirements': ['data_minimization', 'right_to_erasure', 'data_breach_notification'],
                'response_time': 72  # hours for data breaches
            }
        }
        
        self.customer_records = {}
        self.compliance_logs = []
    
    def add_customer(self, customer_id, customer_data):
        """Add customer to compliance system"""
        self.customer_records[customer_id] = {
            'id': customer_id,
            'data': customer_data,
            'compliance_status': {
                'KYC': {'last_completed': None, 'status': 'pending'},
                'AML': {'last_completed': None, 'status': 'monitoring'},
                'GDPR': {'last_completed': None, 'status': 'compliant'}
            },
            'kyc_documents': {},
            'transaction_history': []
        }
    
    def update_kyc_status(self, customer_id):
        """Update KYC compliance status"""
        if customer_id not in self.customer_records:
            return False
        
        customer = self.customer_records[customer_id]
        required_docs = ['government_id', 'address_proof', 'financial_documents']
        
        # Check if all required documents are present
        all_docs_present = all(doc in customer['kyc_documents'] for doc in required_docs)
        
        status = 'complete' if all_docs_present else 'incomplete'
        customer['compliance_status']['KYC'] = {
            'last_completed': datetime.datetime.now(),
            'status': status
        }
        
        self.compliance_logs.append({
            'timestamp': datetime.datetime.now(),
            'customer_id': customer_id,
            'type': 'KYC_update',
            'status': status
        })
        
        return True
    
    def check_aml_rules(self, customer_id, transaction_amount):
        """Check transaction against AML rules"""
        if customer_id not in self.customer_records:
            return False, "Customer not found"
        
        customer = self.customer_records[customer_id]
        
        # Check if transaction exceeds threshold
        aml_threshold = self.regulations['AML']['thresholds']['suspicious_transaction']
        
        if transaction_amount > aml_threshold:
            alert = {
                'timestamp': datetime.datetime.now(),
                'customer_id': customer_id,
                'transaction_amount': transaction_amount,
                'alert_type': 'high_value_transaction',
                'rule_violation': f'Transaction exceeds threshold of {aml_threshold}'
            }
            
            self.compliance_logs.append(alert)
            return True, "Suspicious activity detected"
        
        # Check transaction patterns
        recent_transactions = [
            t for t in customer['transaction_history'] 
            if (datetime.datetime.now() - t['timestamp']).days <= 1
        ]
        
        # Check for rapid successive transactions
        if len(recent_transactions) >= 5:
            daily_total = sum(t['amount'] for t in recent_transactions)
            if daily_total > 15000:
                alert = {
                    'timestamp': datetime.datetime.now(),
                    'customer_id': customer_id,
                    'alert_type': 'transaction_pattern',
                    'rule_violation': 'Suspicious transaction pattern detected'
                }
                
                self.compliance_logs.append(alert)
                return True, "Suspicious transaction pattern detected"
        
        return False, "Transaction compliant"
    
    def add_transaction(self, customer_id, amount, transaction_type='standard'):
        """Add transaction and check AML compliance"""
        if customer_id not in self.customer_records:
            return False
        
        customer = self.customer_records[customer_id]
        
        transaction_record = {
            'id': len(customer['transaction_history']) + 1,
            'amount': amount,
            'type': transaction_type,
            'timestamp': datetime.datetime.now()
        }
        
        customer['transaction_history'].append(transaction_record)
        
        # Check AML rules
        is_suspicious, message = self.check_aml_rules(customer_id, amount)
        
        return not is_suspicious  # Return True if transaction is compliant
    
    def generate_compliance_report(self, customer_id):
        """Generate compliance report for a customer"""
        if customer_id not in self.customer_records:
            return None
        
        customer = self.customer_records[customer_id]
        report = {
            'customer_id': customer_id,
            'as_of_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'compliance_status': customer['compliance_status'],
            'transaction_count': len(customer['transaction_history']),
            'recent_alerts': [
                log for log in self.compliance_logs
                if log['customer_id'] == customer_id
            ][-5:]  # Last 5 compliance logs
        }
        
        return report

# Example usage
compliance_system = ComplianceMonitor()

# Add a customer
compliance_system.add_customer(
    'CUST001',
    {
        'name': 'John Doe',
        'dob': '1985-05-15',
        'address': '123 Main St, Anytown, USA',
        'account_type': 'standard'
    }
)

# Add KYC documents
compliance_system.customer_records['CUST001']['kyc_documents'] = {
    'government_id': 'path/to/id.jpg',
    'address_proof': 'path/to/utility_bill.pdf',
    'financial_documents': 'path/to/bank_statements.pdf'
}

# Update KYC status
compliance_system.update_kyc_status('CUST001')

# Add transactions
compliance_system.add_transaction('CUST001', 5000, 'wire_transfer')
compliance_system.add_transaction('CUST001', 15000, 'check_deposit')  # This should trigger an alert

# Generate compliance report
report = compliance_system.generate_compliance_report('CUST001')
print(f"\nCompliance Report for {report['customer_id']}:")
print(f"  KYC Status: {report['compliance_status']['KYC']['status']}")
print(f"  AML Status: {report['compliance_status']['AML']['status']}")
print(f"  Transaction Count: {report['transaction_count']}")
print(f"  Recent Alerts: {len(report['recent_alerts'])}")
```

## Challenges and Considerations

### 1. Data Quality
Financial data must be accurate, complete, and timely for AI models to be effective.

### 2. Model Risk
AI models in finance need to be carefully validated and monitored to prevent losses.

### 3. Regulatory Compliance
AI systems must comply with financial regulations like MiFID II, Basel III, etc.

### 4. Cybersecurity
Financial AI systems are prime targets for cyber attacks and need robust security.

### 5. Ethical AI
Ensuring fairness and avoiding bias in credit scoring, lending, and investment decisions.

## Future Directions

- **Quantum Computing**: For complex optimization problems in finance
- **Advanced NLP**: For understanding financial news and sentiment
- **Real-time Analytics**: Continuous risk assessment and monitoring
- **Explainable AI**: Making AI decisions more transparent for regulatory compliance

AI in finance continues to evolve rapidly, offering increasingly sophisticated solutions for trading, risk management, customer service, and regulatory compliance.