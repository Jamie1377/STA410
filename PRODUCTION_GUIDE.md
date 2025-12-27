# Production-Grade Infrastructure Guide for STA410

## Overview

This document describes the production-grade infrastructure improvements implemented for the STA410 Stock Prediction System. These changes transform the system from academic code to production-ready software.

## 1. Structured Logging

### Configuration

All logging is now centralized through `stock_prediction/utils/logger.py`:

```python
from stock_prediction.utils.logger import setup_logger

logger = setup_logger(__name__)
logger.info("Application started")
logger.warning("Data quality issue detected")
logger.error("Trade execution failed", exc_info=True)
```

### Log Levels

- **DEBUG**: Detailed diagnostic information (development)
- **INFO**: General informational messages
- **WARNING**: Warning messages for potentially harmful situations
- **ERROR**: Error messages (failures that don't halt the application)
- **CRITICAL**: Critical errors (application-stopping failures)

### Configuration via Environment

```bash
export LOG_LEVEL=INFO
export LOG_DIR=logs
export LOG_MAX_SIZE=10485760  # 10 MB
export LOG_BACKUP_COUNT=10
```

### Log Output

Logs are written to:
- **Console**: Real-time feedback during execution
- **Daily files**: `logs/sta410_YYYYMMDD.log`
- **Error-only files**: `logs/sta410_errors_YYYYMMDD.log` (ERROR and CRITICAL only)
- **Rotation**: Automatic rollover at 10 MB with 10 backups retained

## 2. Custom Exception Hierarchy

All errors now use custom exceptions for fine-grained error handling:

```python
from stock_prediction.utils.exceptions import (
    DataLoadingError,
    DataValidationError,
    ModelTrainingError,
    TradingExecutionError,
    SecretsManagementError
)

try:
    predictor.load_data()
except DataLoadingError as e:
    logger.error(f"Failed to load data: {e}", exc_info=True)
    # Handle gracefully
```

### Exception Types

- `STA410Exception`: Base exception
- `DataLoadingError`: Data source failures
- `DataValidationError`: Invalid data detected
- `FeatureEngineeringError`: Feature calculation failures
- `ModelTrainingError`: Model training failures
- `ModelPredictionError`: Prediction failures
- `ConfigurationError`: Configuration issues
- `SecretsManagementError`: Missing/invalid API keys
- `TradingExecutionError`: Trade execution failures
- `CacheError`: Cache operation failures
- `APIError`: External API call failures

## 3. Secrets Management

### ⚠️ CRITICAL: Never Commit Secrets

API keys must **never** be committed to version control. Use environment variables instead:

```bash
# Set in .env file (NEVER commit to git)
export ALPACA_API_KEY=your_key_here
export ALPACA_SECRET_KEY=your_secret_here
```

### Safe Credential Loading

```python
from stock_prediction.utils.secrets import get_alpaca_credentials, validate_api_credentials

try:
    api_key, secret_key = get_alpaca_credentials()
    validate_api_credentials(api_key, secret_key)
except SecretsManagementError as e:
    logger.error("Missing API credentials", exc_info=True)
    # Graceful degradation
```

### Production Recommendation

Use a secrets manager:
- **AWS**: AWS Secrets Manager or Parameter Store
- **Azure**: Azure Key Vault
- **GCP**: Google Secret Manager
- **On-premise**: HashiCorp Vault

## 4. Configuration Management

### Environment-Based Configuration

Configuration automatically adapts to environment:

```bash
export ENVIRONMENT=production
export LOG_LEVEL=INFO
export PAPER_TRADING=false  # ⚠️ Live trading
export MAX_PORTFOLIO_RISK=0.05
```

### Using Configuration

```python
from stock_prediction.utils.config import get_config

config = get_config()
logger.info(f"Environment: {config.environment}")
logger.info(f"Paper trading: {config.trading.paper_trading}")
logger.info(f"Max portfolio risk: {config.trading.max_portfolio_risk}")
```

### Configuration Hierarchy

1. `.env` file (if present)
2. Environment variables
3. Built-in defaults
4. Validation checks

### Available Settings

See `.env.example` for all available configuration options.

## 5. Data Validation

### Validating Stock Data

```python
from stock_prediction.utils.validation import DataValidator

# Validate OHLCV data
try:
    DataValidator.validate_ohlcv_data(df, symbol="AAPL")
    DataValidator.validate_time_series_continuity(df, symbol="AAPL")
except DataValidationError as e:
    logger.error(f"Data validation failed: {e}")
```

### Automatic Validation on Load

```bash
export VALIDATE_DATA=true  # Automatic validation
```

### Validation Checks

- OHLCV columns present and numeric
- No negative prices or volumes
- High ≥ Low, Close ∈ [Low, High]
- Time series continuity
- NaN/infinite value detection
- Feature variance checks

## 6. Testing Framework

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test category
pytest tests/ -v -m unit

# With coverage
pytest tests/ --cov=stock_prediction --cov-report=html

# Parallel execution
pytest tests/ -n auto
```

### Test Organization

- `tests/test_validation.py`: Data validation tests
- `tests/test_config.py`: Configuration tests
- `tests/conftest.py`: Shared fixtures and configuration

### Test Markers

- `@pytest.mark.unit`: Unit tests (fast)
- `@pytest.mark.integration`: Integration tests (slower)
- `@pytest.mark.slow`: Slow tests (run separately)

## 7. CI/CD Pipeline

### GitHub Actions Workflow

Automated testing on every push:

```bash
.github/workflows/ci-cd.yml
```

### Pipeline Stages

1. **Test**: Run pytest on Python 3.9, 3.10, 3.11
2. **Security**: Check for vulnerabilities with bandit and safety
3. **Build**: Create distribution packages
4. **Docs**: Generate documentation
5. **Notify**: Report results

### Enabling CI/CD

1. Push code to GitHub repository
2. Actions automatically run on push/PR
3. View results in GitHub Actions tab

## 8. Docker Containerization

### Building Docker Image

```bash
docker build -t sta410:latest .
```

### Running with Docker Compose

```bash
# Create .env file with secrets
cp .env.example .env
# Edit .env with real values

# Start services
docker-compose up -d

# View logs
docker-compose logs -f sta410

# Stop services
docker-compose down
```

### What's Included

- **sta410**: Main application
- **redis**: Caching layer (optional)
- **prometheus**: Metrics collection (optional)
- Persistent volumes for logs and cache
- Health checks
- Resource limits (2 CPU, 4 GB RAM)

### Running Specific Command

```bash
docker-compose exec sta410 python -c "from stock_prediction.core import StockPredictor; print('Ready')"
```

## 9. Removed: Warnings Suppression

Previously, all warnings were suppressed:

```python
# OLD (BAD)
import warnings
warnings.filterwarnings('ignore')  # Hides everything!
```

This is now removed. Warnings are displayed during development and logged in production.

## 10. Removed: Hardcoded Secrets

Previously, API keys were hardcoded:

```python
# OLD (SECURITY RISK)
api_key = pd.read_json("API_KEYs.json",orient="index").iloc[0].values[0]
```

Now use environment variables or .env file:

```python
# NEW (SECURE)
from stock_prediction.utils.secrets import get_alpaca_credentials
api_key, secret_key = get_alpaca_credentials()
```

## 11. Production Deployment Checklist

Before deploying to production:

- [ ] **Secrets**: Set ALPACA_API_KEY and ALPACA_SECRET_KEY
- [ ] **Environment**: Set ENVIRONMENT=production
- [ ] **Trading**: Confirm PAPER_TRADING=false only if intentional
- [ ] **Logging**: Set LOG_LEVEL=WARNING for production
- [ ] **Tests**: All tests pass (`pytest tests/ -v`)
- [ ] **Build**: Docker image builds successfully
- [ ] **Configuration**: Run `get_config()` validation
- [ ] **API Credentials**: Validate with `validate_api_credentials()`
- [ ] **Data**: Verify data loading and validation
- [ ] **Monitoring**: Set up log aggregation and alerting

## 12. Monitoring and Observability

### Log Aggregation

Logs can be aggregated with:
- **ELK Stack** (Elasticsearch, Logstash, Kibana)
- **Splunk**
- **CloudWatch** (AWS)
- **Stackdriver** (GCP)

### Metrics Collection

The `prometheus` service in docker-compose.yml enables metrics collection.

### Health Checks

Docker includes health checks:

```bash
docker-compose ps  # Shows health status
```

## 13. Graceful Degradation

The system now includes proper fallback strategies:

```python
try:
    data = load_data_from_api()
except APIError:
    logger.warning("API unavailable, using cached data")
    data = load_cached_data()
    if data is None:
        logger.error("No cached data available, shutting down gracefully")
        sys.exit(1)
```

## 14. Error Recovery

Recovery strategies are implemented for:

- API timeouts: Automatic retry with exponential backoff
- Missing data: Use cached data or skip unavailable symbols
- Model training failures: Log and alert, use previous model
- Trade execution failures: Cancel orders, log for review

## 15. Migration Guide

### Updating Existing Code

Old pattern:
```python
import warnings
warnings.filterwarnings('ignore')

api_key = json.load(open("API_KEYs.json"))["key"]
print(f"Loading data for {symbol}")
try:
    result = some_function()
except Exception as e:
    return None
```

New pattern:
```python
from stock_prediction.utils.logger import setup_logger
from stock_prediction.utils.secrets import get_alpaca_credentials
from stock_prediction.utils.exceptions import DataLoadingError

logger = setup_logger(__name__)

try:
    api_key, secret_key = get_alpaca_credentials()
    logger.info(f"Loading data for {symbol}")
    result = some_function()
except DataLoadingError as e:
    logger.error(f"Failed to load data: {e}", exc_info=True)
    return None
```

## 16. Support and Troubleshooting

### Common Issues

**Missing API credentials:**
```bash
# Set environment variables
export ALPACA_API_KEY=pk_...
export ALPACA_SECRET_KEY=...
```

**Configuration validation error:**
```bash
# Check .env file
cat .env.example  # See valid options
```

**Tests failing:**
```bash
# Run with verbose output
pytest tests/ -vv --tb=long
```

**Docker build fails:**
```bash
# Check Python version
docker build --build-arg PYTHON_VERSION=3.10 -t sta410 .
```

## 17. Next Steps

Priority improvements for further production hardening:

1. **REST API**: Flask/FastAPI endpoints for remote predictions
2. **Database**: PostgreSQL for trade history and metrics
3. **Message Queue**: RabbitMQ/Kafka for async processing
4. **Load Testing**: Locust for performance validation
5. **Disaster Recovery**: Backup/restore procedures
6. **Rate Limiting**: API rate limit handling
7. **Authentication**: User authentication for API
8. **SSL/TLS**: Encrypted communication
9. **Monitoring Dashboard**: Real-time metrics visualization
10. **Alerting**: Automated alerts for failures

---

For questions or issues, refer to logs in `logs/` directory and review GitHub Actions workflow results.
